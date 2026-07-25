"""Command-aware correction, intent classification, and slot extraction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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
    def __init__(
        self,
        catalog: Optional[Sequence[Command]] = None,
        threshold: float = 0.58,
        minimum_margin: float = 0.04,
    ):
        self.catalog = list(catalog or build_default_catalog())
        if not self.catalog:
            raise ValueError("command catalog cannot be empty")
        self.threshold = float(threshold)
        self.minimum_margin = float(minimum_margin)
        self._normalized = [normalize_text(item.text) for item in self.catalog]

    def match(self, text: str) -> CommandMatch:
        normalized = normalize_text(text)
        if not normalized:
            return CommandMatch(text, normalized, "", None, {}, 0.0, 0.0, True)

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

        scored = sorted(
            (
                (_similarity(normalized, candidate), index)
                for index, candidate in enumerate(self._normalized)
            ),
            reverse=True,
        )
        best_score, best_index = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        rejected = best_score < self.threshold or (
            margin < self.minimum_margin and best_score < 0.999
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
    normalized = str(text).strip().lower()
    for source, target in COMMON_CONFUSIONS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()]+", "", normalized)
    return normalized


def edit_distance(left: str, right: str) -> int:
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
    "打开空条": "打开空调",
    "关闭空条": "关闭空调",
    "空条": "空调",
    "车闯": "车窗",
    "车创": "车窗",
    "到行": "导航",
    "倒航": "导航",
    "音像": "音量",
    "风亮": "风量",
    "下一手": "下一首",
    "上一手": "上一首",
}
