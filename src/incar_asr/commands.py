"""Command-aware correction, intent classification, and slot extraction.

Phonetic-aware fuzzy matching for robust in-car voice control:
- Pinyin-based similarity catches homophone errors (the dominant ASR error type in Chinese)
- Expanded confusion pairs cover known error patterns from real in-car recordings
- Keyword-aware fallback rescues commands when full-sentence matching fails
- Built-in pinyin table (~300 chars) provides zero-dependency phonetic matching
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Lightweight built-in pinyin table (no external dependency required)
# Covers ~300 most common characters in in-car voice commands.
# ---------------------------------------------------------------------------
_BUILTIN_PINYIN: Dict[str, str] = {
    # Climate / AC
    "空": "kong", "调": "tiao", "打": "da", "开": "kai", "关": "guan", "闭": "bi",
    "温": "wen", "度": "du", "风": "feng", "量": "liang", "冷": "leng", "热": "re",
    "加": "jia", "除": "chu", "雾": "wu", "霜": "shuang", "内": "nei", "外": "wai",
    "循": "xun", "环": "huan", "座": "zuo", "椅": "yi", "方": "fang", "向": "xiang",
    "盘": "pan", "高": "gao", "低": "di", "一": "yi", "点": "dian", "大": "da",
    "小": "xiao", "档": "dang", "设": "she", "置": "zhi", "为": "wei", "把": "ba",
    "前": "qian", "后": "hou", "挡": "dang", "到": "dao", "了": "le", "吗": "ma",
    "吧": "ba", "呀": "ya", "啦": "la", "啊": "a", "呢": "ne", "的": "de",
    # Numbers
    "零": "ling", "二": "er", "三": "san", "四": "si", "五": "wu", "六": "liu",
    "七": "qi", "八": "ba", "九": "jiu", "十": "shi", "两": "liang", "百": "bai",
    # Windows & doors
    "车": "che", "窗": "chuang", "门": "men", "天": "tian", "左": "zuo", "右": "you",
    "全": "quan", "部": "bu", "所": "suo", "有": "you", "锁": "suo", "定": "ding",
    "解": "jie", "备": "bei", "箱": "xiang", "尾": "wei", "主": "zhu", "驾": "jia",
    "副": "fu", "排": "pai", "半": "ban", "留": "liu", "缝": "feng", "条": "tiao",
    # Music & media
    "播": "bo", "放": "fang", "音": "yin", "乐": "yue", "暂": "zan", "停": "ting",
    "继": "ji", "续": "xu", "下": "xia", "上": "shang", "首": "shou", "歌": "ge",
    "声": "sheng", "曲": "qu", "来": "lai", "听": "ting", "切": "qie", "换": "huan",
    # Phone
    "接": "jie", "电": "dian", "话": "hua", "挂": "gua", "断": "duan", "拒": "ju",
    "免": "mian", "提": "ti", "打": "da", "给": "gei", "联": "lian", "系": "xi",
    "帮": "bang", "我": "wo", "拨": "bo", "号": "hao", "码": "ma",
    # Navigation
    "导": "dao", "航": "hang", "回": "hui", "家": "jia", "去": "qu", "公": "gong",
    "司": "si", "查": "cha", "看": "kan", "路": "lu", "线": "xian", "换": "huan",
    "避": "bi", "速": "su", "带": "dai", "火": "huo", "站": "zhan", "机": "ji",
    "场": "chang", "学": "xue", "校": "xiao", "市": "shi", "中": "zhong", "心": "xin",
    "广": "guang", "科": "ke", "技": "ji", "园": "yuan", "汽": "qi", "酒": "jiu",
    "店": "dian", "银": "yin", "行": "hang", "超": "chao", "电": "dian", "影": "ying",
    "体": "ti", "育": "yu", "馆": "guan", "图": "tu", "书": "shu", "博": "bo",
    "物": "wu", "公": "gong", "景": "jing", "维": "wei", "修": "xiu", "洗": "xi",
    "警": "jing", "交": "jiao", "服": "fu", "务": "wu", "铁": "tie", "最": "zui",
    "近": "jin", "加": "jia", "油": "you", "充": "chong", "停": "ting", "厕": "ce",
    "商": "shang", "餐": "can", "厅": "ting", "入": "ru", "口": "kou", "医": "yi",
    "药": "yao", "民": "min", "动": "dong", "物": "wu", "园": "yuan",
    "林": "lin", "俊": "jun", "杰": "jie", "邓": "deng", "紫": "zi", "棋": "qi",
    "陈": "chen", "奕": "yi", "迅": "xun", "孙": "sun", "燕": "yan", "姿": "zi",
    "王": "wang", "菲": "fei", "五": "wu", "月": "yue", "张": "zhang", "友": "you",
    "薛": "xue", "之": "zhi", "谦": "qian", "毛": "mao", "不": "bu", "易": "yi",
    "刘": "liu", "德": "de", "华": "hua", "李": "li", "荣": "rong", "浩": "hao",
    "许": "xu", "嵩": "song", "汪": "wang", "苏": "su", "泷": "long", "梁": "liang",
    "静": "jing", "茹": "ru", "蔡": "cai", "依": "yi", "陶": "tao", "喆": "zhe",
    "力": "li", "宏": "hong", "韩": "han", "红": "hong",
    # Vehicle controls
    "灯": "deng", "远": "yuan", "近": "jin", "光": "guang", "雨": "yu", "刮": "gua",
    "器": "qi", "双": "shuang", "闪": "shan", "快": "kuai", "慢": "man", "胎": "tai",
    "压": "ya", "剩": "sheng", "余": "yu", "状": "zhuang", "态": "tai", "擦": "ca",
    "水": "shui", "行": "xing", "进": "jin",
    # Actions
    "加": "jia", "减": "jian", "速": "su", "启": "qi", "动": "dong", "取": "qu",
    "消": "xiao", "确": "que", "认": "ren", "返": "fan", "退": "tui", "出": "chu",
    "完": "wan", "成": "cheng", "结": "jie", "束": "shu",
}

# Try to load pypinyin for full pinyin coverage; fall back to built-in table.
_pypinyin_loaded = False
try:
    from pypinyin import lazy_pinyin as _lazy_pinyin

    _pypinyin_loaded = True
except ImportError:
    pass


def _to_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin string (no tones), space-separated."""
    if not text:
        return ""
    if _pypinyin_loaded:
        return " ".join(_lazy_pinyin(text))
    # Fallback: use built-in table for covered chars, keep unknown chars as-is
    result: List[str] = []
    for ch in text:
        result.append(_BUILTIN_PINYIN.get(ch, ch))
    return " ".join(result)


# ---------------------------------------------------------------------------
# Domain-critical terms and their common ASR-error variants
# Each entry maps a term to a list of common misrecognition variants.
# ---------------------------------------------------------------------------
_DOMAIN_TERM_VARIANTS: Dict[str, List[str]] = {
    "空调": ["空条", "空跳", "孔调", "控调", "空吊"],
    "车窗": ["车闯", "车创", "车床", "车窗"],
    "导航": ["到行", "倒航", "道航", "导行", "到航"],
    "风量": ["风亮", "风凉", "风两", "风辆", "封量", "丰量"],
    "音量": ["音像", "音亮", "音凉", "阴量", "音两"],
    "温度": ["温都", "文度", "闻度", "温渡"],
    "加热": ["家热", "佳热", "加乐", "加热"],
    "座椅": ["坐椅", "做椅", "座已", "座以", "作椅"],
    "除雾": ["出雾", "处雾", "除务", "除物", "除误"],
    "内循环": ["内巡环", "内循还", "内寻环", "内旬环"],
    "外循环": ["外巡环", "外循还", "外寻环", "外旬环"],
    "后备箱": ["后备相", "后备乡", "后辈箱", "后背箱"],
    "天窗": ["天闯", "天创", "天床", "添窗"],
    "方向盘": ["方像盘", "方向般", "方相盘", "方向旁"],
    "雨刮器": ["雨刮气", "雨挂器", "雨瓜器", "雨刮起"],
    "双闪": ["双山", "双善", "双扇", "霜闪"],
    "胎压": ["胎呀", "太压", "台压", "胎鸭"],
    "免提": ["面提", "棉提", "免题", "免体"],
    "挂断": ["瓜断", "刮断", "挂段", "挂短"],
    "拒接": ["句接", "具接", "据接", "巨接"],
    "播放": ["波放", "博放", "拨放", "播方"],
    "暂停": ["占停", "赞停", "暂庭", "暂听"],
    "下一首": ["下一手", "下一守", "下衣首", "夏一首"],
    "上一首": ["上一手", "上一守", "上衣首", "伤一首"],
}


@dataclass(frozen=True)
class Command:
    text: str
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandMatch:
    raw_text: str
    normalized_text: str
    corrected_text: str
    intent: Optional[str]
    slots: Dict[str, Any]
    confidence: float
    margin: float
    rejected: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommandMatcher:
    """Fuzzy command matcher with phonetic awareness for Chinese ASR errors.

    Matching strategy (tried in order):
    1. Regex-based dynamic matching (temperature, phone, navigation patterns)
    2. Pinyin-aware combined similarity against the command catalog
    3. Keyword-based fallback when full-sentence confidence is marginal

    The combined similarity blends character-level edit distance and
    pinyin-level edit distance, weighting pinyin more heavily because
    Chinese ASR errors are predominantly homophone substitutions.
    """

    def __init__(
        self,
        catalog: Optional[Sequence[Command]] = None,
        threshold: float = 0.52,
        minimum_margin: float = 0.03,
        pinyin_weight: float = 0.6,
    ):
        self.catalog = list(catalog or build_default_catalog())
        if not self.catalog:
            raise ValueError("command catalog cannot be empty")
        self.threshold = float(threshold)
        self.minimum_margin = float(minimum_margin)
        self.pinyin_weight = float(pinyin_weight)
        self._normalized = [normalize_text(item.text) for item in self.catalog]
        # Pre-compute pinyin for all catalog entries (expensive, do once)
        self._pinyin = [_to_pinyin(n) for n in self._normalized]

    def match(self, text: str) -> CommandMatch:
        normalized = normalize_text(text)
        if not normalized:
            return CommandMatch(text, normalized, "", None, {}, 0.0, 0.0, True)

        # 1. Try dynamic (regex) matching first — temperature, phone, navigation
        dynamic = _match_dynamic_command(normalized)
        if dynamic is not None:
            return CommandMatch(
                raw_text=text,
                normalized_text=normalized,
                corrected_text=dynamic.text,
                intent=dynamic.intent,
                slots=dict(dynamic.slots),
                confidence=1.0,
                margin=1.0,
                rejected=False,
            )

        # 2. Apply domain-term fuzzy correction before catalog matching
        corrected_normalized = _fuzzy_correct_terms(normalized)

        # 3. Combined character + pinyin similarity against catalog
        input_pinyin = _to_pinyin(corrected_normalized)
        scored = sorted(
            (
                (
                    _combined_similarity(
                        corrected_normalized,
                        input_pinyin,
                        candidate,
                        self._pinyin[idx],
                        self.pinyin_weight,
                    ),
                    idx,
                )
                for idx, candidate in enumerate(self._normalized)
            ),
            reverse=True,
        )
        best_score, best_index = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        rejected = best_score < self.threshold or (
            margin < self.minimum_margin and best_score < 0.999
        )

        # 4. Keyword-aware fallback: if rejected but keywords overlap, rescue it
        if rejected:
            keyword_match = _keyword_fallback_match(
                corrected_normalized, self.catalog, self._normalized
            )
            if keyword_match is not None:
                kw_text, kw_cmd, kw_score = keyword_match
                return CommandMatch(
                    raw_text=text,
                    normalized_text=normalized,
                    corrected_text=kw_cmd.text,
                    intent=kw_cmd.intent,
                    slots=dict(kw_cmd.slots),
                    confidence=kw_score,
                    margin=0.0,
                    rejected=False,
                )

        if rejected:
            return CommandMatch(
                text, normalized, normalized, None, {}, best_score, margin, True
            )
        command = self.catalog[best_index]
        return CommandMatch(
            raw_text=text,
            normalized_text=normalized,
            corrected_text=command.text,
            intent=command.intent,
            slots=dict(command.slots),
            confidence=best_score,
            margin=margin,
            rejected=False,
        )


def normalize_text(text: str) -> str:
    """Normalize text for matching: apply known corrections, strip punctuation."""
    normalized = str(text).strip().lower()
    for source, target in COMMON_CONFUSIONS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[\s，。！？、,.!?;；:：\"'""''（）()]+", "", normalized)
    return normalized


def edit_distance(left: str, right: str) -> int:
    """Levenshtein edit distance between two strings (character-level)."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _fuzzy_correct_terms(text: str) -> str:
    """Replace known ASR-error variants of domain terms with correct terms.

    Scans the text for each variant and replaces with the canonical term.
    Longer terms are checked first to avoid partial replacements.
    """
    result = text
    # Sort by term length descending so "内循环" is corrected before "循环"
    for term, variants in sorted(
        _DOMAIN_TERM_VARIANTS.items(), key=lambda kv: -len(kv[0])
    ):
        for variant in sorted(variants, key=len, reverse=True):
            if variant in result:
                result = result.replace(variant, term)
    return result


def _combined_similarity(
    char_text: str,
    pinyin_text: str,
    char_candidate: str,
    pinyin_candidate: str,
    pinyin_weight: float = 0.6,
) -> float:
    """Blend character-level and pinyin-level similarity.

    Chinese ASR errors are predominantly homophone substitutions, so pinyin
    similarity is weighted more heavily (default 0.6) than character similarity.
    """
    char_sim = _similarity(char_text, char_candidate)
    pinyin_sim = _similarity(pinyin_text, pinyin_candidate)
    return (1.0 - pinyin_weight) * char_sim + pinyin_weight * pinyin_sim


def _keyword_fallback_match(
    text: str,
    catalog: List[Command],
    normalized_catalog: List[str],
) -> Optional[tuple]:
    """When full-sentence similarity fails, try keyword-overlap matching.

    Extracts 2-4 character substrings from the ASR text, checks which catalog
    entries share the most keyword overlap, and returns the best match if the
    overlap is strong enough.
    """
    if len(text) < 2:
        return None

    # Extract all 2-4 char substrings as "keywords"
    keywords = set()
    for length in (2, 3, 4):
        for i in range(len(text) - length + 1):
            sub = text[i : i + length]
            if sub:
                keywords.add(sub)

    if not keywords:
        return None

    best_score = 0.0
    best_idx = -1
    for idx, candidate in enumerate(normalized_catalog):
        if len(candidate) < 2:
            continue
        # Count how many keywords from the text appear in this candidate
        hits = sum(1 for kw in keywords if kw in candidate)
        # Also check the reverse: candidate substrings in text
        for length in (2, 3):
            for i in range(len(candidate) - length + 1):
                if candidate[i : i + length] in text:
                    hits += 0.5
        # Normalize by candidate length
        score = hits / max(len(candidate), 1)
        if score > best_score:
            best_score = score
            best_idx = idx

    # Require a minimum keyword overlap to accept
    if best_score >= 0.25 and best_idx >= 0:
        return (text, catalog[best_idx], min(best_score, 0.75))
    return None


def build_default_catalog() -> List[Command]:
    """Build more than 200 deterministic in-car command variants."""
    commands: List[Command] = [
        Command("打开空调", "climate.open"),
        Command("关闭空调", "climate.close"),
        Command("温度调高一点", "climate.temperature_up"),
        Command("温度调低一点", "climate.temperature_down"),
        Command("风量调大一点", "climate.fan_up"),
        Command("风量调小一点", "climate.fan_down"),
        Command("打开内循环", "climate.recirculation", {"mode": "internal"}),
        Command("打开外循环", "climate.recirculation", {"mode": "external"}),
        Command("打开前挡风除雾", "climate.defrost", {"position": "front"}),
        Command("打开后挡风除雾", "climate.defrost", {"position": "rear"}),
        Command("播放音乐", "music.play"),
        Command("暂停音乐", "music.pause"),
        Command("继续播放", "music.resume"),
        Command("下一首", "music.next"),
        Command("上一首", "music.previous"),
        Command("音量调大", "music.volume_up"),
        Command("音量调小", "music.volume_down"),
        Command("接听电话", "phone.answer"),
        Command("挂断电话", "phone.hangup"),
        Command("拒接电话", "phone.reject"),
        Command("打开免提", "phone.speaker", {"enabled": True}),
        Command("关闭免提", "phone.speaker", {"enabled": False}),
        Command("导航回家", "navigation.start", {"destination": "家"}),
        Command("导航去公司", "navigation.start", {"destination": "公司"}),
        Command("关闭导航", "navigation.stop"),
        Command("查看路线", "navigation.route_overview"),
        Command("换一条路线", "navigation.reroute"),
        Command("避开高速", "navigation.avoid", {"road_type": "高速"}),
    ]

    for temperature in range(16, 33):
        chinese = number_to_chinese(temperature)
        commands.extend(
            [
                Command(
                    f"温度调到{chinese}度",
                    "climate.set_temperature",
                    {"temperature": temperature},
                ),
                Command(
                    f"空调设置为{chinese}度",
                    "climate.set_temperature",
                    {"temperature": temperature},
                ),
            ]
        )

    for level in range(1, 8):
        chinese = number_to_chinese(level)
        commands.extend(
            [
                Command(
                    f"风量调到{chinese}档",
                    "climate.set_fan",
                    {"level": level},
                ),
                Command(
                    f"空调风量{chinese}档",
                    "climate.set_fan",
                    {"level": level},
                ),
            ]
        )

    window_positions = {
        "左前": "front_left",
        "右前": "front_right",
        "左后": "rear_left",
        "右后": "rear_right",
        "全部": "all",
    }
    for label, value in window_positions.items():
        commands.extend(
            [
                Command(f"打开{label}车窗", "window.open", {"position": value}),
                Command(f"关闭{label}车窗", "window.close", {"position": value}),
                Command(
                    f"{label}车窗开一半",
                    "window.set_percentage",
                    {"position": value, "percentage": 50},
                ),
                Command(
                    f"{label}车窗留一条缝",
                    "window.set_percentage",
                    {"position": value, "percentage": 10},
                ),
            ]
        )
    commands.extend(
        [
            Command("打开天窗", "sunroof.open"),
            Command("关闭天窗", "sunroof.close"),
            Command(
                "天窗开一半", "sunroof.set_percentage", {"percentage": 50}
            ),
        ]
    )

    destinations = [
        "最近的加油站",
        "最近的充电站",
        "最近的停车场",
        "最近的医院",
        "最近的药店",
        "最近的厕所",
        "最近的商场",
        "最近的餐厅",
        "最近的高速入口",
        "火车站",
        "机场",
        "学校",
        "市中心",
        "人民广场",
        "科技园",
        "汽车站",
        "酒店",
        "银行",
        "超市",
        "电影院",
        "体育馆",
        "图书馆",
        "博物馆",
        "公园",
        "景区",
        "维修店",
        "洗车店",
        "交警大队",
        "高速服务区",
        "地铁站",
    ]
    for destination in destinations:
        commands.extend(
            [
                Command(
                    f"导航到{destination}",
                    "navigation.start",
                    {"destination": destination},
                ),
                Command(
                    f"带我去{destination}",
                    "navigation.start",
                    {"destination": destination},
                ),
            ]
        )

    contacts = [
        "妈妈",
        "爸爸",
        "老婆",
        "老公",
        "姐姐",
        "哥哥",
        "弟弟",
        "妹妹",
        "张三",
        "李四",
        "王五",
        "赵六",
        "刘老师",
        "陈经理",
        "客服",
        "保险公司",
        "道路救援",
        "公司前台",
        "同事小王",
        "同事小李",
        "医生",
        "班主任",
        "物业",
        "酒店前台",
        "餐厅",
        "家里",
        "秘书",
        "助理",
        "朋友小陈",
        "朋友小刘",
    ]
    for contact in contacts:
        commands.extend(
            [
                Command(f"打电话给{contact}", "phone.call", {"contact": contact}),
                Command(f"帮我联系{contact}", "phone.call", {"contact": contact}),
            ]
        )

    artists = [
        "周杰伦",
        "林俊杰",
        "邓紫棋",
        "陈奕迅",
        "孙燕姿",
        "王菲",
        "五月天",
        "张学友",
        "薛之谦",
        "毛不易",
        "刘德华",
        "李荣浩",
        "张杰",
        "许嵩",
        "汪苏泷",
        "梁静茹",
        "蔡依林",
        "陶喆",
        "王力宏",
        "韩红",
    ]
    for artist in artists:
        commands.extend(
            [
                Command(f"播放{artist}的歌", "music.play_artist", {"artist": artist}),
                Command(f"来一首{artist}", "music.play_artist", {"artist": artist}),
            ]
        )

    # Short-form variants for common commands (improves matching on short ASR output)
    commands.extend(
        [
            Command("风量调大", "climate.fan_up"),
            Command("风量调小", "climate.fan_down"),
            Command("音量调大", "music.volume_up"),
            Command("音量调小", "music.volume_down"),
            Command("温度调高", "climate.temperature_up"),
            Command("温度调低", "climate.temperature_down"),
            Command("打开车窗", "window.open", {"position": "all"}),
            Command("关闭车窗", "window.close", {"position": "all"}),
            Command("打开车门", "door.open_all"),
            Command("关闭车门", "door.close_all"),
        ]
    )

    vehicle_actions = {
        "打开车灯": "vehicle.light_on",
        "关闭车灯": "vehicle.light_off",
        "打开远光灯": "vehicle.high_beam_on",
        "关闭远光灯": "vehicle.high_beam_off",
        "打开近光灯": "vehicle.low_beam_on",
        "关闭近光灯": "vehicle.low_beam_off",
        "打开雾灯": "vehicle.fog_light_on",
        "关闭雾灯": "vehicle.fog_light_off",
        "打开雨刮器": "vehicle.wiper_on",
        "关闭雨刮器": "vehicle.wiper_off",
        "雨刮器快一点": "vehicle.wiper_up",
        "雨刮器慢一点": "vehicle.wiper_down",
        "打开双闪": "vehicle.hazard_on",
        "关闭双闪": "vehicle.hazard_off",
        "打开后备箱": "vehicle.trunk_open",
        "关闭后备箱": "vehicle.trunk_close",
        "查看胎压": "vehicle.query_tire_pressure",
        "查看剩余油量": "vehicle.query_fuel",
        "查看剩余电量": "vehicle.query_battery",
        "查看车辆状态": "vehicle.query_status",
        "座椅加热": "vehicle.seat_heating_on",
        "关闭座椅加热": "vehicle.seat_heating_off",
        "方向盘加热": "vehicle.steering_heating_on",
        "关闭方向盘加热": "vehicle.steering_heating_off",
    }
    commands.extend(Command(text, intent) for text, intent in vehicle_actions.items())

    unique: Dict[str, Command] = {}
    for command in commands:
        unique.setdefault(normalize_text(command.text), command)
    catalog = list(unique.values())
    if len(catalog) < 200:
        raise AssertionError(f"default command catalog unexpectedly has only {len(catalog)} items")
    return catalog


def write_catalog(path: Path, catalog: Optional[Sequence[Command]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(command) for command in (catalog or build_default_catalog())]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_catalog(path: Path) -> List[Command]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("command catalog must be a JSON list")
    return [
        Command(
            text=str(item["text"]),
            intent=str(item["intent"]),
            slots=dict(item.get("slots", {})),
        )
        for item in data
    ]


def number_to_chinese(value: int) -> str:
    digits = "零一二三四五六七八九"
    if 0 <= value < 10:
        return digits[value]
    if 10 <= value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    if 20 <= value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    return str(value)


def chinese_to_number(text: str) -> Optional[int]:
    normalized = text.strip()
    if normalized.isdigit():
        return int(normalized)
    digits = {char: index for index, char in enumerate("零一二三四五六七八九")}
    if normalized == "十":
        return 10
    if "十" in normalized:
        left, right = normalized.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if normalized in digits:
        return digits[normalized]
    return None


def _match_dynamic_command(normalized: str) -> Optional[Command]:
    temperature = re.fullmatch(r"(?:把)?(?:空调)?温度(?:设置为|设为|调到)([零一二三四五六七八九十\d]+)度", normalized)
    if temperature:
        value = chinese_to_number(temperature.group(1))
        if value is not None and 16 <= value <= 32:
            return Command(
                f"温度调到{number_to_chinese(value)}度",
                "climate.set_temperature",
                {"temperature": value},
            )
    call = re.fullmatch(r"(?:打电话给|帮我联系)(.+)", normalized)
    if call and call.group(1):
        contact = call.group(1)
        return Command(f"打电话给{contact}", "phone.call", {"contact": contact})
    navigate = re.fullmatch(r"(?:导航到|带我去)(.+)", normalized)
    if navigate and navigate.group(1):
        destination = navigate.group(1)
        return Command(
            f"导航到{destination}", "navigation.start", {"destination": destination}
        )
    return None


def _similarity(left: str, right: str) -> float:
    longest = max(len(left), len(right), 1)
    return 1.0 - edit_distance(left, right) / longest


COMMON_CONFUSIONS: Mapping[str, str] = {
    # === 空调 / AC ===
    "打开空条": "打开空调",
    "关闭空条": "关闭空调",
    "空条": "空调",
    "打开空跳": "打开空调",
    "关闭空吊": "关闭空调",
    "打开孔调": "打开空调",
    "打空调": "打开空调",
    # === 车窗 / Windows ===
    "车闯": "车窗",
    "车创": "车窗",
    "车床": "车窗",
    "车昌": "车窗",
    "打开车闯": "打开车窗",
    "关闭车闯": "关闭车窗",
    # === 导航 / Navigation ===
    "到行": "导航",
    "倒航": "导航",
    "道航": "导航",
    "导行": "导航",
    "到航": "导航",
    "导杭": "导航",
    "到行去": "导航去",
    "到行到": "导航到",
    # === 音乐 / Music ===
    "下一手": "下一首",
    "上一手": "上一首",
    "下衣首": "下一首",
    "上衣首": "上一首",
    "暂停音乐": "暂停音乐",
    "播发音乐": "播放音乐",
    "波放": "播放",
    "博放": "播放",
    "暂庭": "暂停",
    "占停": "暂停",
    # === 音量/风量 ===
    "音像": "音量",
    "阴量": "音量",
    "音两": "音量",
    "风亮": "风量",
    "风凉": "风量",
    "风两": "风量",
    "封量": "风量",
    # === 温度 ===
    "温都": "温度",
    "文度": "温度",
    "闻度": "温度",
    "温渡": "温度",
    "温度调高": "温度调高",
    "温度条高": "温度调高",
    # === 循环 ===
    "内寻环": "内循环",
    "内巡环": "内循环",
    "内循还": "内循环",
    "外寻环": "外循环",
    "外巡环": "外循环",
    "外循还": "外循环",
    # === 除雾/除霜 ===
    "出雾": "除雾",
    "处雾": "除雾",
    "除务": "除雾",
    "除物": "除雾",
    "除霜": "除霜",
    "出霜": "除霜",
    "处霜": "除霜",
    # === 座椅加热 ===
    "坐椅": "座椅",
    "做椅": "座椅",
    "作椅": "座椅",
    "座已": "座椅",
    "座以": "座椅",
    # === 电话 ===
    "接听电话": "接听电话",
    "结听电话": "接听电话",
    "接听电化": "接听电话",
    "挂段": "挂断",
    "瓜断": "挂断",
    "刮断": "挂断",
    "据接": "拒接",
    "句接": "拒接",
    "巨接": "拒接",
    # === 天窗 ===
    "天闯": "天窗",
    "天创": "天窗",
    "添窗": "天窗",
    "打开天闯": "打开天窗",
    # === 后备箱 ===
    "后背箱": "后备箱",
    "后备相": "后备箱",
    "后辈箱": "后备箱",
    # === 车辆控制 ===
    "双山": "双闪",
    "双善": "双闪",
    "双扇": "双闪",
    "胎呀": "胎压",
    "太压": "胎压",
    "台压": "胎压",
    "鱼刮器": "雨刮器",
    "雨挂器": "雨刮器",
    "雨瓜起": "雨刮器",
    "方像盘": "方向盘",
    "方向般": "方向盘",
    "方相盘": "方向盘",
    # === 通用 ===
    "打开": "打开",
    "打开开": "打开",
    "开启": "打开",
    "起开": "打开",
    "打凯": "打开",
    "关闭": "关闭",
    "关必": "关闭",
    "观闭": "关闭",
    "关掉": "关闭",
    "调到": "调到",
    "条到": "调到",
    "调到到": "调到",
    "条岛": "调到",
    "换一条": "换一条",
    "换一挑": "换一条",
    "换一跳": "换一条",
    "带我去": "带我去",
    "带我取": "带我去",
    "带我去去": "带我去",
    "打电话": "打电话",
    "打电化": "打电话",
    "打点话": "打电话",
    "打垫话": "打电话",
}
