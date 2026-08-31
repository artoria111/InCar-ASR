"""Car-ASR Cockpit V2 — Liquid4All 车载座舱 + sherpa-onnx ASR 后端
前端: Liquid4All cookbook audio-car-cockpit (MIT License)
后端: sherpa-onnx Paraformer-zh-small INT8 + 车载指令理解
"""
import numpy as np, torch, torchaudio, onnxruntime as ort, soundfile as sf
import time, io, os, json, subprocess, tempfile, base64, re
from flask import Flask, request, jsonify, send_from_directory

MODEL = '/root/work/car-asr-engine/model/model.int8.onnx'
TOKENS = '/root/work/car-asr-engine/model/sherpa_tokens.txt'
CMVN = '/root/work/car-asr-engine/model/sherpa_cmvn.npz'
STATIC = '/root/work/car-asr-engine/static'

import warnings; warnings.filterwarnings("ignore")
os.environ["ORT_DISABLE_CPU_INFO"] = "1"

print('[Init] Loading ASR model...')
sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
tokens_raw = open(TOKENS).read().strip().split('\n')
id2token = {i: parts[0] for i, line in enumerate(tokens_raw) if (parts:=line.split())}
cmvn = np.load(CMVN); shift = torch.tensor(cmvn['means']); scale = torch.tensor(cmvn['vars'])
print('[Init] Ready')

app = Flask(__name__)

# ============================================================

# Chinese numeral to int converter

_CN_NUMS = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"两":2}
def _cn2int(s):
    s = s.strip()
    if s.isdigit(): return int(s)
    if s in _CN_NUMS: return _CN_NUMS[s]
    if "十" in s:
        parts = s.split("十")
        tens = _CN_NUMS.get(parts[0], 1) if parts[0] else 1
        ones = _CN_NUMS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return 26  # fallback


# ============================================================
# Fuzzy ASR correction layer — fixes homophone errors before parsing
# ============================================================

_FUZZY_MAP = {
    # AC / climate
    "空条":"空调","空跳":"空调","孔调":"空调","控调":"空调","空吊":"空调",
    "打空调":"打开空调","打开空条":"打开空调","关闭空条":"关闭空调",
    # Windows
    "车闯":"车窗","车创":"车窗","车床":"车窗","车昌":"车窗",
    # Navigation
    "到行":"导航","倒航":"导航","道航":"导航","导行":"导航","到航":"导航","导杭":"导航",
    # Music
    "下一手":"下一首","上一手":"上一首","下衣首":"下一首","上衣首":"上一首",
    "波放":"播放","博放":"播放","拨放":"播放","播方":"播放",
    "占停":"暂停","赞停":"暂停","暂庭":"暂停","暂听":"暂停",
    # Volume / fan
    "音像":"音量","阴量":"音量","音两":"音量","音亮":"音量","音凉":"音量",
    "风亮":"风量","风凉":"风量","风两":"风量","封量":"风量","丰量":"风量",
    # Temperature
    "温都":"温度","文度":"温度","闻度":"温度","温渡":"温度",
    # Circulation
    "内寻环":"内循环","内巡环":"内循环","内循还":"内循环",
    "外寻环":"外循环","外巡环":"外循环","外循还":"外循环",
    # Defrost
    "出雾":"除雾","处雾":"除雾","除务":"除雾","除物":"除雾","除误":"除雾",
    "出霜":"除霜","处霜":"除霜",
    # Seats
    "坐椅":"座椅","做椅":"座椅","作椅":"座椅","座已":"座椅","座以":"座椅",
    "家热":"加热","佳热":"加热","加乐":"加热",
    # Phone
    "瓜断":"挂断","刮断":"挂断","挂段":"挂断","挂短":"挂断",
    "据接":"拒接","句接":"拒接","具接":"拒接","巨接":"拒接",
    "结听":"接听","接听电化":"接听电话",
    "面提":"免提","免题":"免提",
    "打电化":"打电话","打点话":"打电话","打垫话":"打电话",
    # Sunroof
    "天闯":"天窗","天创":"天窗","添窗":"天窗",
    # Trunk
    "后背箱":"后备箱","后备相":"后备箱","后辈箱":"后备箱","后背乡":"后备箱",
    # Vehicle
    "双山":"双闪","双善":"双闪","双扇":"双闪",
    "胎呀":"胎压","太压":"胎压","台压":"胎压",
    "鱼刮器":"雨刮器","雨挂器":"雨刮器","雨瓜起":"雨刮器",
    "方像盘":"方向盘","方向般":"方向盘","方相盘":"方向盘",
    # General
    "打开开":"打开","起开":"打开","打凯":"打开",
    "关必":"关闭","观闭":"关闭","关掉":"关闭",
    "条到":"到","条岛":"到",
    "换一挑":"换一条","换一跳":"换一条",
    "带我取":"带我去","带我去去":"带我去",
    # 温度数字 ASR 错字（"都" 误识别为 "度"）
    "二十六都":"二十六度","二十七都":"二十七度","二十八都":"二十八度",
    "二十九都":"二十九度","三十都":"三十度","三十一都":"三十一度",
    "三十二都":"三十二度","十六都":"十六度","十七都":"十七度",
    "十八都":"十八度","十九都":"十九度","二十都":"二十度",
    "二十一都":"二十一度","二十二都":"二十二度","二十三都":"二十三度",
    "二十四都":"二十四度","二十五都":"二十五度",
}

_FUZZY_ORDERED = sorted(_FUZZY_MAP.items(), key=lambda kv: -len(kv[0]))

def fuzzy_correct(text):
    """Apply domain-specific ASR error correction before command parsing."""
    result = text.replace(" ", "")
    for wrong, correct in _FUZZY_ORDERED:
        if wrong in result:
            result = result.replace(wrong, correct)
    return result


# ============================================================
# Car command parser (maps ASR text -> cockpit actions)
# ============================================================
def parse_command(text: str) -> dict:
    """Parse ASR text into car cockpit commands"""
    t = fuzzy_correct(text)
    cmd = {"text": text, "actions": [], "corrected": t}

    # Seat heating (must come before temperature '热' check)
    if '座椅加热' in t:
        if '打开' in t or '开启' in t:
            cmd["actions"].append({"type": "climate", "target": "seat_heat", "value": True})
        elif '关闭' in t:
            cmd["actions"].append({"type": "climate", "target": "seat_heat", "value": False})

    # Climate — temperature rules (re-ordered: explicit up/down first, then '热' -> down, '冷' -> up)
    if m := re.search(r'温度.*?([\d一二三四五六七八九十两百]+)度', t):
        cmd["actions"].append({"type": "climate", "target": "temperature", "value": _cn2int(m.group(1))})
    elif '温度调高' in t or '温度高一点' in t:
        cmd["actions"].append({"type": "climate", "target": "temperature", "delta": 1})
    elif '温度调低' in t or '温度低一点' in t:
        cmd["actions"].append({"type": "climate", "target": "temperature", "delta": -1})
    elif '热' in t and '加热' not in t and '座椅' not in t:
        cmd["actions"].append({"type": "climate", "target": "temperature", "delta": -1})
    elif '冷' in t or '冻' in t:
        cmd["actions"].append({"type": "climate", "target": "temperature", "delta": 1})

    if '风量' in t:
        if m := re.search(r'([\d一二三四五六七八九十两百]+)档', t):
            cmd["actions"].append({"type": "climate", "target": "fan_speed", "value": _cn2int(m.group(1))})
        elif '大' in t:
            cmd["actions"].append({"type": "climate", "target": "fan_speed", "delta": 1})
        elif '小' in t:
            cmd["actions"].append({"type": "climate", "target": "fan_speed", "delta": -1})

    if '打开空调' in t: cmd["actions"].append({"type": "climate", "target": "ac", "value": True})
    if '关闭空调' in t: cmd["actions"].append({"type": "climate", "target": "ac", "value": False})
    if '内循环' in t: cmd["actions"].append({"type": "climate", "target": "circulation", "value": True})
    if '外循环' in t: cmd["actions"].append({"type": "climate", "target": "circulation", "value": False})
    if '除雾' in t or '除霜' in t:
        if '关闭' in t:
            cmd["actions"].append({"type": "climate", "target": "defrost", "value": False})
        else:
            cmd["actions"].append({"type": "climate", "target": "defrost", "value": True})

    # Windows
    if ('打开车窗' in t or '开车窗' in t) and not any(p in t for p in ['左前','右前','左后','右后','所有','全部']):
        cmd["actions"].append({"type": "window", "target": "all", "action": "open"})
    if ('关闭车窗' in t or '关车窗' in t) and not any(p in t for p in ['左前','右前','左后','右后','所有','全部']):
        cmd["actions"].append({"type": "window", "target": "all", "action": "close"})
    if '打开所有车窗' in t or '车窗全开' in t:
        cmd["actions"].append({"type": "window", "target": "all", "action": "open"})
    if '关闭所有车窗' in t:
        cmd["actions"].append({"type": "window", "target": "all", "action": "close"})

    for w in ['左前','右前','左后','右后']:
        if w in t:
            w_map = {'左前': 'fl', '右前': 'fr', '左后': 'rl', '右后': 'rr'}
            action = "open" if '打开' in t else "close"
            cmd["actions"].append({"type": "window", "target": w_map[w], "action": action})

    # Doors & locks
    if '锁车' in t or '锁定车门' in t:
        cmd["actions"].append({"type": "door", "target": "lock", "value": True})
    if '解锁' in t or '开车门' in t:
        cmd["actions"].append({"type": "door", "target": "lock", "value": False})
    # Individual door open
    door_map = {'左前':'fl','主驾':'fl','右前':'fr','副驾':'fr',
                '左后':'rl','右后':'rr','后排':'rear'}
    if '所有车门' in t or '全部车门' in t:
        action = "close" if ('关闭' in t or '关' in t) else "open"
        cmd["actions"].append({"type": "door", "target": "all", "action": action})
    elif '车门' in t and ('打开' in t or '开' in t):
        for key, val in door_map.items():
            if key in t:
                cmd["actions"].append({"type": "door", "target": val, "action": "open"})
                break
    if '后备箱' in t or '尾门' in t:
        action = "open" if '打开' in t else "close"
        cmd["actions"].append({"type": "door", "target": "trunk", "action": action})

    if '天窗' in t:
        action = "open" if '打开' in t else "close"
        cmd["actions"].append({"type": "window", "target": "sunroof", "action": action})

    # Phone
    if '挂断' in t: cmd["actions"].append({"type": "phone", "target": "hangup"})
    if '拒接' in t: cmd["actions"].append({"type": "phone", "target": "reject"})
    if '接听' in t: cmd["actions"].append({"type": "phone", "target": "answer"})
    if '免提' in t:
        cmd["actions"].append({"type": "phone", "target": "speaker", "value": '关闭' not in t})

    # Vehicle controls
    if '车灯' in t:
        cmd["actions"].append({"type": "vehicle", "target": "lights", "value": '打开' in t or '关闭' not in t})
    if '双闪' in t:
        cmd["actions"].append({"type": "vehicle", "target": "hazard_lights", "value": '打开' in t or '关闭' not in t})
    if '雨刮' in t:
        cmd["actions"].append({"type": "vehicle", "target": "wipers", "value": '打开' in t or '关闭' not in t})
    if '方向盘加热' in t:
        cmd["actions"].append({"type": "climate", "target": "steering_heat", "value": '打开' in t or '关闭' not in t})

    # Music
    if '下一首' in t:
        cmd["actions"].append({"type": "media", "target": "next"})
    elif '上一首' in t:
        cmd["actions"].append({"type": "media", "target": "previous"})
    elif '继续播放' in t:
        cmd["actions"].append({"type": "media", "target": "play", "value": True})
    elif '播放' in t:
        cmd["actions"].append({"type": "media", "target": "play", "value": True})
    if '暂停' in t or '关闭音乐' in t:
        cmd["actions"].append({"type": "media", "target": "play", "value": False})
    if '音量大' in t or '大声' in t or '音量调大' in t:
        cmd["actions"].append({"type": "media", "target": "volume", "delta": 10})
    if '音量小' in t or '小声' in t or '音量调小' in t:
        cmd["actions"].append({"type": "media", "target": "volume", "delta": -10})

    # Navigation
    if '导航' in t or '回家' in t or '去' in t:
        cmd["actions"].append({"type": "nav", "target": "navigate", "destination": t})

    # Speed simulation
    if '加速' in t: cmd["actions"].append({"type": "vehicle", "target": "speed", "delta": 20})
    if '减速' in t: cmd["actions"].append({"type": "vehicle", "target": "speed", "delta": -20})
    if '停车' in t: cmd["actions"].append({"type": "vehicle", "target": "speed", "value": 0})

    return cmd


# ============================================================
# ASR inference
# ============================================================
def decode_audio(audio_bytes):
    try: audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32'); return audio, sr
    except: pass
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as tmp: tp=tmp.name
        with tempfile.NamedTemporaryFile(suffix='.bin',delete=False) as inp:
            inp.write(audio_bytes); ip=inp.name
        subprocess.run(['ffmpeg','-y','-i',ip,'-ar','16000','-ac','1','-sample_fmt','s16',tp],capture_output=True,timeout=30)
        audio,sr=sf.read(tp,dtype='float32'); os.unlink(tp); os.unlink(ip); return audio,sr
    except: raise ValueError("Cannot decode")

def run_asr(audio_bytes):
    audio, sr = decode_audio(audio_bytes)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    dur = len(audio) / max(sr, 1)
    t0 = time.time()
    waveform = torch.tensor(audio).unsqueeze(0) * (1 << 15)
    if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, frame_length=25,
        frame_shift=10, dither=0.0, energy_floor=0.0, window_type='hamming',
        sample_frequency=16000, snip_edges=True)
    lfr_list = []; t = 0
    while t + 7 <= fbank.shape[0]: lfr_list.append(fbank[t:t+7].flatten()); t += 6
    if not lfr_list: return {"text": "", "error": "too short"}
    lfr = torch.stack(lfr_list)
    lfr = ((lfr + shift) * scale).numpy().astype(np.float32)
    lens = np.array([lfr.shape[0]], dtype=np.int32)
    logits, tok_num = sess.run(None, {'speech': lfr[np.newaxis, :, :], 'speech_lengths': lens})
    n_tok = min(int(tok_num[0]), 40)
    best = np.argmax(logits[0, :n_tok, :], axis=1)
    text = ''.join(id2token.get(int(tid), '') for tid in best
                   if int(tid) not in (0, 1, 2, 3) and int(tid) in id2token)
    elapsed = time.time() - t0
    return {"text": text, "delay_ms": round(elapsed * 1000), "rtf": round(elapsed / dur, 4), "duration_s": round(dur, 2)}


# ============================================================
# Routes
# ============================================================
@app.route("/")
def root():
    return send_from_directory(STATIC, 'index.html')

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC, filename)

@app.route("/api/recognize", methods=["POST"])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files['file']
    try:
        result = run_asr(f.read())
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    # Attach cockpit commands
    result["command"] = parse_command(result.get("text", ""))
    return jsonify(result)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Car-ASR Cockpit V2")
    print("  http://0.0.0.0:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)