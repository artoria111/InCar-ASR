# 模型压缩报告

> 生成时间: 2026-07-17 14:55

---

## 1. 压缩策略

采用**二级压缩**策略:

```
Paraformer-large (220M)
    ↓ 知识蒸馏 (温度 T=4.0, KL+MSE+Attention)
Paraformer-tiny (5.2M)
    ↓ INT8 PTQ (AMCT)
OM 离线模型 (< 50MB)
```

## 2. 各阶段对比

| 阶段 | 参数量 | 模型体积 | CER (AISHELL-1) | 推理速度 |
|------|:------:|:------:|:-----:|:------:|
| Teacher (Paraformer-large) | 220M | 881MB | 1.95% | RTF 0.025 |
| Student (Paraformer-tiny) | 5.2M | 21MB | — | RTF 0.004 |
| INT8 Quantized | 5.2M | <50MB* | — | RTF <0.1* |

> *INT8 量化需在昇腾 AMCT 工具链上完成（当前开发板内存不足，需交叉编译）

## 3. 知识蒸馏配置

| 超参数 | 值 | 说明 |
|------|:--:|------|
| 温度 T | 4.0 | 软化概率分布 |
| KL 权重 | 0.5 | Logits 蒸馏 |
| 特征权重 | 0.3 | 隐藏层 MSE |
| 注意力权重 | 0.2 | 注意力矩阵 |
| 优化器 | AdamW | lr=1e-4, wd=1e-5 |
| 训练轮数 | 20 | Cosine LR |

## 4. ONNX 精度校验

| 指标 | 值 | 说明 |
|------|:--:|------|
| 余弦相似度 | > 0.999 | PyTorch vs ONNX 输出 |
| 最大绝对误差 | < 1e-3 | 逐元素对比 |
| 是否通过 | ✅ | 精度无损 |

## 5. 下一步

1. ASCEND AMCT INT8 PTQ 量化（需交叉编译环境）
2. ATC 转换为 OM 离线模型
3. NPU 端侧推理性能 Profiling
