#!/usr/bin/env python3
"""
车载场景测试方案 — 多维度测试矩阵生成器

测试维度:
  噪声类型 × SNR × 指令类别 × 车速场景

生成 ≥ 1000 个测试用例，输出 JSONL 格式。

Usage:
  python 02_test_matrix.py --output test_plan.json
"""

import argparse, json, os, itertools, random


# ============================================================
# 测试维度定义
# ============================================================
NOISE_TYPES = [
    "engine_idle", "engine_acceleration", "road_asphalt", "road_concrete",
    "tire_noise", "wind_noise", "air_conditioner", "rain",
    "horn_and_traffic", "cabin_music", "cabin_speech", "window_open",
]

SNR_LEVELS = [0, 5, 10, 15, 20]  # dB

COMMAND_CATEGORIES = {
    "导航": ["导航到最近的加油站", "导航回家", "导航去公司", "打开导航", "关闭导航",
             "查看路线", "前面堵不堵", "换一条路线", "避开高速", "还有多久到"],
    "空调": ["打开空调", "关闭空调", "温度调到二十六度", "温度调高一点", "温度调低一点",
             "风量调大一点", "风量调小一点", "打开内循环", "打开外循环", "打开前挡风除雾"],
    "电话": ["打电话给张三", "打电话给妈妈", "接听电话", "挂断电话", "拒接电话",
             "打开免提", "关闭免提", "回拨刚才的电话", "重拨上一个电话", "查看通话记录"],
    "音乐": ["播放音乐", "暂停音乐", "下一首", "上一首", "音量调大",
             "音量调小", "播放周杰伦的歌", "播放排行榜", "打开收音机", "切换到蓝牙音乐"],
    "车窗": ["打开车窗", "关闭车窗", "车窗开一半", "打开天窗", "关闭天窗",
             "打开左前车窗", "打开右前车窗", "打开后排车窗", "关闭所有车窗", "车窗留一条缝"],
    "车辆控制": ["启动车辆", "打开车灯", "关闭车灯", "打开远光灯", "打开雨刮器",
               "关闭雨刮器", "查看车辆状态", "查看胎压", "查看剩余油量", "打开后备箱"],
}

SPEED_SCENARIOS = [
    {"speed": "parked", "label": "停车 (0 km/h)"},
    {"speed": "city_low", "label": "城市低速 (1-30 km/h)"},
    {"speed": "city_mid", "label": "城市中速 (31-60 km/h)"},
    {"speed": "highway", "label": "高速 (>60 km/h)"},
]


def generate_test_cases(num_cases: int = 1000) -> list:
    """生成测试用例矩阵"""
    cases = []

    # 笛卡尔积: 噪声 × SNR × 指令 × 车速
    combinations = list(itertools.product(
        NOISE_TYPES, SNR_LEVELS, list(COMMAND_CATEGORIES.keys()), SPEED_SCENARIOS
    ))

    # 随机采样
    random.seed(42)
    if len(combinations) > num_cases:
        combinations = random.sample(combinations, num_cases)

    for noise, snr, cat, speed in combinations:
        command = random.choice(COMMAND_CATEGORIES[cat])
        cases.append({
            "test_id": f"TC{len(cases)+1:04d}",
            "noise_type": noise,
            "snr_db": snr,
            "category": cat,
            "command": command,
            "speed_scenario": speed["speed"],
            "expected": command,
            "status": "pending",
            "cer": None,
            "rtf": None,
            "delay_ms": None,
        })

    return cases


def generate_test_plan(cases: list, output_path: str):
    """生成测试计划 Markdown"""
    from collections import Counter

    noise_counts = Counter(c["noise_type"] for c in cases)
    snr_counts = Counter(c["snr_db"] for c in cases)
    cat_counts = Counter(c["category"] for c in cases)

    plan = f"""# 车载语音识别系统测试方案

> 测试用例总数: **{len(cases)}**
> 项目目标: ≥ 1000 个测试用例

---

## 1. 测试矩阵

### 测试维度

| 维度 | 取值 |
|------|------|
| 噪声类型 | {len(NOISE_TYPES)} 类 |
| 信噪比 SNR | {SNR_LEVELS} dB |
| 指令类别 | {len(COMMAND_CATEGORIES)} 类 |
| 车速场景 | {len(SPEED_SCENARIOS)} 种 |

### 噪声类型分布

| 噪声 | 用例数 |
|------|:--:|
"""
    for noise, count in noise_counts.most_common():
        plan += f"| {noise} | {count} |\n"

    plan += """
### SNR 分布

| SNR | 用例数 |
|------|:--:|
"""
    for snr in sorted(snr_counts):
        plan += f"| {snr} dB | {snr_counts[snr]} |\n"

    plan += """
### 指令类别分布

| 类别 | 用例数 |
|------|:--:|
"""
    for cat, count in cat_counts.most_common():
        plan += f"| {cat} | {count} |\n"

    plan += f"""
## 2. 评估指标

| 指标 | 目标值 | 评估方法 |
|------|:--:|------|
| CER (安静) | < 3% | 无噪声条件下 |
| CER (噪声) | < 8% | SNR 5-15dB |
| RTF | < 0.1 | Ascend 310B 实测 |
| 端到端延迟 | < 500ms | 音频结束→文本输出 |
| VAD 准确率 | > 95% | 噪声环境下的语音/非语音分类 |

## 3. 测试流程

1. 准备测试音频: 干净语音 × 噪声混合 (augment_audio.py)
2. 批量推理: 01_e2e_system.py --mode batch
3. CER 统计: 按维度分组生成报告
4. Bad case 分析: 导出高错误率样本
5. 最终报告: 输出 Markdown + JSON

## 4. 测试用例清单

测试用例保存在 `test_cases.jsonl`，每行一个 JSON 对象。
"""
    # Save plan
    plan_path = output_path.replace(".json", "_plan.md")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan)

    # Save cases
    cases_path = output_path.replace(".json", ".jsonl")
    with open(cases_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Test plan: {plan_path}")
    print(f"Test cases: {cases_path} ({len(cases)} cases)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="modules/05_system_integration_testing/outputs/test_plan.json")
    parser.add_argument("--num_cases", type=int, default=1200)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cases = generate_test_cases(args.num_cases)
    generate_test_plan(cases, args.output)


if __name__ == "__main__":
    main()
