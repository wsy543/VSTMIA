# 联邦学习防御方法 - 参数配置说明

## 概览

本项目实现了三种联邦学习防御方法，通过 `--defence` 参数切换：
1. **dpsgd** - 差分隐私随机梯度下降 (DP-SGD)
2. **mixupmmd** - Mixup 数据增强 + MMD 正则化
3. **l2** - L2 正则化

---

## 1. DP-SGD (差分隐私随机梯度下降)

### 原理
对每个客户端的梯度进行 **裁剪 (clip)** 后添加 **高斯噪声**，防止攻击者从模型参数或梯度中推断出个体用户的隐私信息。

### 关键参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `--defence dpsgd` | - | - | 启用 DP-SGD 防御 |
| `--dp_clip_norm` | 100.0 | 1.0 ~ 100.0 | 梯度 L2 范数裁剪阈值。值越小隐私保护越强，但收敛越慢 |
| `--dp_noise_multiplier` | 0.001 | 0.0 ~ 1.0 | 噪声乘数，实际噪声标准差 = noise_multiplier × clip_norm |

### ⚠️ 收敛说明
- 当使用较小的 `clip_norm`(如 1.0~5.0)时，梯度被严重裁剪，需要 **100 轮以上** 训练才能收敛
- 默认值 `clip_norm=100, noise=0.001` 几乎不影响梯度，主要用于验证代码正确性
- **如需真正隐私保护**，请调低 `clip_norm` 并同时增加 `--training_round` 至 200+

### 使用示例

```bash
# 代码验证模式 (几乎不影响收敛)
python main.py --dataset CIFAR10 --model resnet \
  --model_path ./models_defence --train_model True \
  --defence dpsgd --dp_clip_norm 100.0 --dp_noise_multiplier 0.001 \
  --training_round 50 --epochs 2 --lr 0.005

# 中等隐私保护 (需足够多训练轮数)
python main.py --dataset CIFAR10 --model resnet \
  --model_path ./models_defence --train_model True \
  --defence dpsgd --dp_clip_norm 10.0 --dp_noise_multiplier 0.01 \
  --training_round 100 --epochs 2 --lr 0.005

# 强隐私保护 (收敛慢，需大量轮数)
python main.py --dataset CIFAR10 --model resnet \
  --model_path ./models_defence --train_model True \
  --defence dpsgd --dp_clip_norm 1.0 --dp_noise_multiplier 0.5 \
  --training_round 200 --epochs 5 --lr 0.001
```

---

## 2. MixupMMD (Mixup 数据增强 + MMD 正则化)

### 原理
- **Mixup**: 将不同样本按比例混合，增强模型泛化能力
- **MMD (Maximum Mean Discrepancy)**: 最小化原始特征与 Mixup 后特征的分布差异，防止模型记住单个样本细节

### 关键参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `--defence mixupmmd` | - | - | 启用 MixupMMD 防御 |
| `--mixup_alpha` | 0.2 | 0.1 ~ 1.0 | Beta 分布参数。值越小 Mixup 越"极端"，值越大混合越均匀 |
| `--mmd_lambda` | 0.01 | 0.001 ~ 0.1 | MMD 正则化权重。值越大对特征分布的约束越强 |

### 测试结果 (CIFAR10, ResNet, 5轮)
- Round 1: 客户端 Acc ~19%, 全局 Acc ~13%
- Round 2: 客户端 Acc ~22%, 全局 Acc ~26%
- Round 3: 客户端 Acc ~27%, 全局 Acc ~40%
- 收敛速度略慢于基线（Mixup 增加训练难度），但无错误

### 使用示例

```bash
python main.py --dataset CIFAR10 --model resnet \
  --model_path ./models_defence --train_model True \
  --defence mixupmmd --mixup_alpha 0.2 --mmd_lambda 0.01 \
  --training_round 50 --epochs 2 --lr 0.005
```

### 注意事项
- Mixup 修改了损失计算方式，会降低训练准确率但提升泛化能力
- MMD 部分从模型倒数第二层提取特征；若 MMD 计算失败(日志有 warning)，Mixup 部分仍正常工作

---

## 3. L2 正则化

### 原理
在优化器中增加 weight_decay（权重衰减），对模型参数施加 L2 惩罚，防止模型过拟合。

### 关键参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `--defence l2` | - | - | 启用 L2 正则化防御 |
| `--l2_lambda` | 0.001 | 0.0001 ~ 0.01 | L2 正则化系数 (即 optimizer 的 weight_decay) |

### 测试结果 (CIFAR10, ResNet, 5轮)
- Round 2: 客户端 Acc ~37%, 全局 Acc ~28% ✅
- Round 3: 客户端 Acc ~44%, 全局 Acc ~48% ✅
- Round 4: 客户端 Acc ~52%, 全局 Acc ~55% ✅
- **与基线几乎完全一致**，适合作为轻量级防御

### 使用示例

```bash
python main.py --dataset CIFAR10 --model resnet \
  --model_path ./models_defence --train_model True \
  --defence l2 --l2_lambda 0.001 \
  --training_round 50 --epochs 2 --lr 0.005
```

---

## 通用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--training_round` | 200 | 联邦学习总轮数 |
| `--epochs` | 2 | 每轮每个客户端本地训练 epoch 数 |
| `--lr` | 0.005 | 学习率 (ResNet), MobileNet 建议 0.001 |
| `--batch_size` | 64 | 批大小 |
| `--client_num` | 5 | 客户端总数 |
| `--participant` | 5 | 每轮参与训练的客户端数 |
| `--model_path` | ./models_main | 模型保存路径（防御模式请改为 ./models_defence） |

---

## 防御效果对比 (CIFAR10, ResNet, 5轮, batch_size=32)

| 防御方法 | Round 4 全局 Acc | 收敛速度影响 |
|----------|-----------------|-------------|
| 无防御 (baseline) | ~55% | 基准 |
| L2 (λ=0.001) | ~55% | 几乎无影响 ✅ |
| MixupMMD (α=0.2, λ=0.01) | ~40% (Round 3) | 略慢 ⚠️ |
| DP-SGD (clip=100, noise=0.001) | ~10% (局部 26%) | 需更多轮数 ⚠️ |
| DP-SGD (clip=5, noise=0.05) | ~10% | 需要 ≥100 轮训练 🔴 |

> **建议**: 使用 `defence=l2` 作为首选防御（零成本），`defence=mixupmmd` 作为增强防御，
> `defence=dpsgd` 需配合大量训练轮数使用。
