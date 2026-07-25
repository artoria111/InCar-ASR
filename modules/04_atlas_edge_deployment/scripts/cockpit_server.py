#!/usr/bin/env python3
"""Car-ASR Cockpit Server — 车载仿真座舱 + 语音控制"""
import numpy as np, torch, torchaudio, onnxruntime as ort, soundfile as sf
import time, io, os, json, subprocess, tempfile
from flask import Flask, request, jsonify

MODEL = '/root/work/car-asr-engine/model/model.int8.onnx'
TOKENS = '/root/work/car-asr-engine/model/sherpa_tokens.txt'
CMVN = '/root/work/car-asr-engine/model/sherpa_cmvn.npz'

import warnings; warnings.filterwarnings("ignore")
os.environ["ORT_DISABLE_CPU_INFO"] = "1"

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
tokens_raw = open(TOKENS).read().strip().split('\n')
id2token = {i: parts[0] for i, line in enumerate(tokens_raw) if (parts:=line.split())}
cmvn = np.load(CMVN); shift = torch.tensor(cmvn['means']); scale = torch.tensor(cmvn['vars'])

app = Flask(__name__)

COCKPIT_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Car-ASR Cockpit</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0d14;color:#c0c8d8;overflow:hidden;height:100vh}
.screen{display:flex;height:100vh}
/* ===== LEFT: Dashboard ===== */
.dash{flex:6;background:linear-gradient(180deg,#0d1117 0%,#0a0d14 100%);position:relative;display:flex;flex-direction:column;justify-content:center;align-items:center;border-right:1px solid rgba(0,200,255,.08)}
.speed-ring{position:relative;width:280px;height:280px}
.speed-ring canvas{position:absolute;top:0;left:0}
.speed-val{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
.speed-val .num{font-size:4em;font-weight:200;color:#fff;line-height:1;text-shadow:0 0 30px rgba(0,200,255,.3)}
.speed-val .unit{font-size:.8em;color:#556;letter-spacing:2px}
.rpm-bar{width:240px;height:4px;background:rgba(255,255,255,.1);border-radius:2px;margin:20px 0;overflow:hidden}
.rpm-bar .fill{height:100%;background:linear-gradient(90deg,#00c8ff,#00ff88,#ffcc00,#ff4444);transition:width .3s}
.gear{font-size:2em;color:#00c8ff;font-weight:200;margin-top:10px}
/* ===== RIGHT: Center Console ===== */
.console{flex:4;display:flex;flex-direction:column;background:#111620;padding:20px}
.console h2{color:#00c8ff;font-size:1em;letter-spacing:2px;margin-bottom:16px;text-transform:uppercase}
/* Voice area */
.voice-area{flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;border:1px solid rgba(0,200,255,.1);border-radius:16px;padding:20px;margin-bottom:12px;background:rgba(0,0,0,.2)}
.mic-ring{width:80px;height:80px;border-radius:50%;border:2px solid rgba(0,200,255,.3);display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .3s;margin-bottom:12px}
.mic-ring:hover{border-color:rgba(0,200,255,.6)}
.mic-ring.active{border-color:#00c8ff;box-shadow:0 0 30px rgba(0,200,255,.2);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 10px rgba(0,200,255,.1)}50%{box-shadow:0 0 40px rgba(0,200,255,.3)}}
.mic-ring .icon{font-size:2em}
.voice-text{font-size:1.3em;color:#e0e8f0;text-align:center;min-height:36px;word-break:break-all;max-width:100%}
.voice-status{font-size:.7em;color:#556;margin-top:4px}
/* Climate panel */
.climate{display:flex;justify-content:space-around;align-items:center;padding:12px;background:rgba(0,0,0,.2);border-radius:12px;margin-bottom:12px}
.climate .item{text-align:center}
.climate .label{font-size:.6em;color:#556;text-transform:uppercase;margin-top:4px}
.climate .val{font-size:1.2em;color:#fff;font-weight:200}
.climate .arrow{cursor:pointer;color:#00c8ff55;font-size:1.5em;transition:color .2s;user-select:none}
.climate .arrow:hover{color:#00c8ff}
/* Feature indicators */
.features{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.feat{padding:8px 12px;background:rgba(0,0,0,.2);border-radius:8px;text-align:center;font-size:.7em;transition:all .3s}
.feat.on{background:rgba(0,200,255,.15);color:#00c8ff;box-shadow:0 0 10px rgba(0,200,255,.1)}
.feat.off{color:#556}
.feat .icon{display:block;font-size:1.4em;margin-bottom:2px}
/* Upload */
.upload-btn{font-size:.65em;color:#00c8ff55;text-align:center;cursor:pointer;margin-top:8px}
.upload-btn:hover{color:#00c8ff}
.upload-btn input{display:none}
</style></head><body>
<div class="screen">
<!-- DASHBOARD -->
<div class="dash">
<div class="speed-ring"><canvas id="speedCanvas" width="280" height="280"></canvas>
<div class="speed-val"><div class="num" id="speedVal">0</div><div class="unit">km/h</div></div></div>
<div class="rpm-bar"><div class="fill" id="rpmFill" style="width:15%"></div></div>
<div class="gear" id="gear">P</div>
</div>
<!-- CENTER CONSOLE -->
<div class="console">
<h2>◈ CAR·ASR Cockpit</h2>
<div class="voice-area">
<div class="mic-ring" id="micRing" onclick="toggleMic()"><span class="icon">🎤</span></div>
<div class="voice-text" id="voiceText">说出车载指令</div>
<div class="voice-status" id="voiceStatus">就绪</div>
</div>
<div class="climate">
<div class="arrow" onclick="adjTemp(-1)">◀</div>
<div class="item"><div class="val" id="tempVal">26</div><div class="label">温度 °C</div></div>
<div class="arrow" onclick="adjTemp(1)">▶</div>
<div style="width:20px"></div>
<div class="arrow" onclick="adjFan(-1)">◀</div>
<div class="item"><div class="val" id="fanVal">3</div><div class="label">风量</div></div>
<div class="arrow" onclick="adjFan(1)">▶</div>
</div>
<div class="features">
<div class="feat off" id="featAC"><span class="icon">❄️</span>空调</div>
<div class="feat off" id="featCirc"><span class="icon">🔄</span>内循环</div>
<div class="feat off" id="featDef"><span class="icon">🌫</span>除雾</div>
<div class="feat off" id="featMusic"><span class="icon">🎵</span>音乐</div>
<div class="feat off" id="featNav"><span class="icon">🧭</span>导航</div>
<div class="feat off" id="featWindow"><span class="icon">🪟</span>车窗</div>
</div>
<label class="upload-btn">📁 上传音频 <input type="file" accept="audio/*" onchange="uploadFile(this.files[0])"></label>
</div>
</div>
<script>
// === Speedometer ===
const canvas=document.getElementById('speedCanvas'),ctx=canvas.getContext('2d');
function drawSpeed(v){
 ctx.clearRect(0,0,280,280); const cx=140,cy=140,r=120;
 // Arc background
 ctx.beginPath(); ctx.arc(cx,cy,r-10,0.75*Math.PI,2.25*Math.PI);
 ctx.strokeStyle='rgba(255,255,255,.08)'; ctx.lineWidth=16; ctx.stroke();
 // Arc value
 const angle=0.75*Math.PI+(Math.min(v,160)/160)*1.5*Math.PI;
 ctx.beginPath(); ctx.arc(cx,cy,r-10,0.75*Math.PI,angle);
 const g=ctx.createLinearGradient(0,0,280,0);
 g.addColorStop(0,'#00c8ff'); g.addColorStop(.5,'#00ff88'); g.addColorStop(1,'#ff4444');
 ctx.strokeStyle=g; ctx.lineWidth=16; ctx.stroke();
 // Ticks
 for(let i=0;i<=160;i+=20){
  const a=0.75*Math.PI+(i/160)*1.5*Math.PI;
  const x1=cx+(r-22)*Math.cos(a),y1=cy+(r-22)*Math.sin(a);
  const x2=cx+(r-8)*Math.cos(a),y2=cy+(r-8)*Math.sin(a);
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
  ctx.strokeStyle='rgba(255,255,255,.3)'; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle='#556'; ctx.font='9px system-ui';
  ctx.fillText(i,x1-10,y1+14);
 }
}
drawSpeed(0);

// === Mic ===
let mr,chunks=[],recording=false, temp=26, fan=3, speed=0;
function toggleMic(){
 const ring=document.getElementById('micRing'), txt=document.getElementById('voiceText');
 if(recording){mr.stop();recording=false;ring.classList.remove('active');document.getElementById('voiceStatus').textContent='识别中...';return}
 navigator.mediaDevices.getUserMedia({audio:{}}).then(s=>{
  mr=new MediaRecorder(s);chunks=[];
  mr.ondataavailable=e=>chunks.push(e.data);
  mr.onstop=()=>{s.getTracks().forEach(t=>t.stop());sendAudio(new Blob(chunks,{type:'audio/webm'}))};
  mr.start();recording=true;ring.classList.add('active');txt.textContent='聆听中...';document.getElementById('voiceStatus').textContent='录音中';
 }).catch(()=>{txt.textContent='麦克风未授权'});
}
function uploadFile(f){
 if(!f)return;document.getElementById('voiceText').textContent='识别中...';document.getElementById('voiceStatus').textContent='处理中...';
 const d=new FormData();d.append('file',f);
 fetch('/api/recognize',{method:'POST',body:d}).then(r=>r.json()).then(applyCommand).catch(()=>{});
}
function sendAudio(blob){
 const d=new FormData();d.append('file',blob,'r.webm');
 fetch('/api/recognize',{method:'POST',body:d}).then(r=>r.json()).then(applyCommand).catch(()=>{});
}
function applyCommand(d){
 const txt=document.getElementById('voiceText'),st=document.getElementById('voiceStatus');
 txt.textContent=d.text||'(空)';st.textContent=(d.delay_ms||0)+'ms RTF '+(d.rtf||0).toFixed(3);
 processCommand(d.text||'');
}
function processCommand(t){
 // Parse commands and update cockpit
 if(t.includes('温度')||t.includes('调')){
  if(t.includes('高')||t.includes('大'))adjTemp(1);
  else if(t.includes('低')||t.includes('小'))adjTemp(-1);
  const m=t.match(/(\d+)度/);if(m)temp=parseInt(m[1]);
  document.getElementById('tempVal').textContent=temp;
 }
 if(t.includes('风')||t.includes('风量')){
  if(t.includes('大'))adjFan(1);
  else if(t.includes('小'))adjFan(-1);
  const m=t.match(/(\d+)档/);if(m)fan=parseInt(m[1]);
  document.getElementById('fanVal').textContent=fan;
 }
 if(t.includes('打开空调')||t.includes('空调')){toggleFeat('featAC',true)}
 if(t.includes('关闭空调')){toggleFeat('featAC',false)}
 if(t.includes('内循环')){toggleFeat('featCirc',true)}
 if(t.includes('外循环')){toggleFeat('featCirc',false)}
 if(t.includes('除雾')||t.includes('除霜')){toggleFeat('featDef',true)}
 if(t.includes('导航')){toggleFeat('featNav',!!t.match(/导航|回家|去/))}
 if(t.includes('音乐')||t.includes('播放')){toggleFeat('featMusic',!!t.match(/音乐|播放|首/))}
 if(t.includes('车窗')||t.includes('天窗')){toggleFeat('featWindow',!!t.match(/打开/))}
 // Speed simulation
 if(t.includes('加速')||t.includes('快')){speed=Math.min(160,speed+20);updateSpeed()}
 if(t.includes('减速')||t.includes('慢')){speed=Math.max(0,speed-20);updateSpeed()}
 if(t.includes('停')){speed=0;document.getElementById('gear').textContent='P';updateSpeed()}
 if(t.includes('出发')||t.includes('开车')||t.includes('导航')){document.getElementById('gear').textContent='D'}
}
function adjTemp(d){temp=Math.max(16,Math.min(32,temp+d));document.getElementById('tempVal').textContent=temp}
function adjFan(d){fan=Math.max(1,Math.min(7,fan+d));document.getElementById('fanVal').textContent=fan}
function toggleFeat(id,on){const el=document.getElementById(id);el.className='feat '+(on?'on':'off')}
function updateSpeed(){document.getElementById('speedVal').textContent=speed;drawSpeed(speed);document.getElementById('rpmFill').style.width=(15+speed/160*70)+'%'}
</script></body></html>"""

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

@app.route("/")
def root(): return COCKPIT_HTML

@app.route("/api/recognize", methods=["POST"])
def recognize():
    if 'file' not in request.files: return jsonify({"error":"no file"}),400
    f=request.files['file']; audio_bytes=f.read()
    try: audio,sr=decode_audio(audio_bytes)
    except: return jsonify({"error":"format"}),400
    if audio.ndim>1: audio=audio.mean(axis=1)
    if len(audio)/max(sr,1)<0.3: return jsonify({"error":"too short"}),400
    dur=len(audio)/sr; t0=time.time()
    waveform=torch.tensor(audio).unsqueeze(0)*(1<<15)
    if sr!=16000: waveform=torchaudio.transforms.Resample(sr,16000)(waveform)
    fbank=torchaudio.compliance.kaldi.fbank(waveform,num_mel_bins=80,frame_length=25,frame_shift=10,
        dither=0.0,energy_floor=0.0,window_type='hamming',sample_frequency=16000,snip_edges=True)
    lfr=[]; t=0
    while t+7<=fbank.shape[0]: lfr.append(fbank[t:t+7].flatten()); t+=6
    lfr=torch.stack(lfr) if lfr else torch.zeros(1,560)
    lfr=((lfr+shift)*scale).numpy().astype(np.float32)
    lens=np.array([lfr.shape[0]],dtype=np.int32)
    logits,tok_num=sess.run(None,{'speech':lfr[np.newaxis,:,:],'speech_lengths':lens})
    n_tok=min(int(tok_num[0]),40); best=np.argmax(logits[0,:n_tok,:],axis=1)
    text=''.join(id2token.get(int(tid),'') for tid in best if int(tid) not in(0,1,2,3) and int(tid) in id2token)
    elapsed=time.time()-t0
    return jsonify({"text":text,"duration_s":round(dur,2),"delay_ms":round(elapsed*1000),"rtf":round(elapsed/dur,4)})

if __name__=="__main__":
    import ssl,sys
    if not os.path.exists('/tmp/car-asr.crt'):
        os.system('openssl req -x509 -newkey rsa:2048 -keyout /tmp/car-asr.key -out /tmp/car-asr.crt -days 365 -nodes -subj "/CN=cockpit" 2>/dev/null')
    print("\n"+"="*50)
    print("  Car-ASR Cockpit Server")
    print("  https://0.0.0.0:5443")
    print("="*50+"\n")
    app.run(host="0.0.0.0",port=5443,debug=False,ssl_context=('/tmp/car-asr.crt','/tmp/car-asr.key'))
