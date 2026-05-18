"""
联邦学习防御模块
支持的防御方法:
1. DP-SGD: 差分隐私随机梯度下降
2. MixupMMD: Mixup 数据增强 + MMD 正则化
3. L2: L2 正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)


class DefenceManager:
    """
    防御管理器, 在客户端本地训练时自动应用所选防御
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.defence = getattr(args, 'defence', 'none').lower()
        self._last_print_epoch = -1  # 记录上次打印的 epoch，避免每个 batch 都打印

        # DP-SGD 参数
        #   注意: noise_std = noise_multiplier × clip_norm
        #   grad_norm 很大 (~50) 是因为跨所有参数的总范数,
        #   而噪声是每参数独立加的, 需控制 noise_std << 每参数平均梯度
        #   默认 noise_std = 0.0005 × 60 = 0.03 (远小于原版 0.1)
        self.dp_clip_norm = getattr(args, 'dp_clip_norm', 60.0)
        self.dp_noise_multiplier = getattr(args, 'dp_noise_multiplier', 0.0005)

        # MixupMMD 参数
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.2)
        self.mmd_lambda = getattr(args, 'mmd_lambda', 0.01)

        # L2 参数
        self.l2_lambda = getattr(args, 'l2_lambda', 0.001)

        logger.info(f"[Defence] 初始化防御方法: {self.defence}")
        if self.defence == 'dpsgd':
            logger.info(f"  DP-SGD: clip_norm={self.dp_clip_norm}, noise_multiplier={self.dp_noise_multiplier}")
        elif self.defence == 'mixupmmd':
            logger.info(f"  MixupMMD: mixup_alpha={self.mixup_alpha}, mmd_lambda={self.mmd_lambda}")
        elif self.defence == 'l2':
            logger.info(f"  L2: lambda={self.l2_lambda}")

    def on_before_forward(self, model, data, label):
        """
        前向传播之前的预处理.
        用于 MixupMMD: 对数据和标签进行 Mixup 增强.
        Returns: (processed_data, processed_label, extra_info)
        """
        if self.defence == 'mixupmmd' and self.mixup_alpha > 0:
            # Mixup 数据增强
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(data.device)
            mixed_data = lam * data + (1 - lam) * data[index, :]
            label_a, label_b = label, label[index]
            return mixed_data, (label_a, label_b, lam), {'mixup_index': index, 'lam': lam}
        return data, label, {}

    def on_after_loss(self, model, data, label, outputs, loss, extra_info):
        """
        计算损失后、反向传播前的操作.
        用于 MixupMMD: 如果使用了 Mixup, 需要修正损失计算.
        Returns: 修正后的 loss
        """
        if self.defence == 'mixupmmd' and extra_info.get('mixup_index') is not None:
            lam = extra_info['lam']
            label_a, label_b, _ = label if isinstance(label, tuple) else (label, label, 1.0)
            loss_fn = nn.CrossEntropyLoss()
            loss = lam * loss_fn(outputs, label_a) + (1 - lam) * loss_fn(outputs, label_b)

            # MMD 正则化: 对特征进行分布对齐
            if self.mmd_lambda > 0:
                try:
                    features = self._extract_features(model, data)
                    if features is not None:
                        index = extra_info['mixup_index']
                        feat_a = features
                        feat_b = features[index]
                        mmd_val = self._mmd_rbf(feat_a, feat_b)
                        loss = loss + self.mmd_lambda * mmd_val
                except Exception as e:
                    logger.warning(f"[Defence] MMD 计算失败 (非致命): {e}")

        return loss

    def on_after_backward(self, model, epoch=None):
        """
        反向传播后的梯度处理.
        用于 DP-SGD: 梯度裁剪 + 添加高斯噪声.
        """
        if self.defence == 'dpsgd':
            self._apply_dpsgd(model, epoch=epoch)

    def get_optimizer_kwargs(self):
        """
        获取优化器额外参数.
        用于 L2: 向优化器传递 weight_decay.
        """
        if self.defence == 'l2':
            return {'weight_decay': self.l2_lambda}
        return {}

    def _apply_dpsgd(self, model, epoch=None):
        """
        DP-SGD: 逐层裁剪梯度 + 添加高斯噪声.
        """
        # === Step 1: 计算总梯度范数 ===
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 每个 epoch 只打印一次，避免刷屏
        if epoch is not None and epoch != self._last_print_epoch:
            self._last_print_epoch = epoch
            print(f"[DP-SGD] Epoch {epoch + 1}: grad_norm={total_norm:.2f}, "
                  f"clip_norm={self.dp_clip_norm}, noise_std={self.dp_noise_multiplier * self.dp_clip_norm:.4f}")

        # === Step 2: 梯度裁剪 ===
        clip_coef = min(1.0, self.dp_clip_norm / (total_norm + 1e-8))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        # === Step 3: 添加高斯噪声 ===
        if self.dp_noise_multiplier > 0:
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=self.dp_noise_multiplier * self.dp_clip_norm,
                        size=p.grad.shape,
                        device=p.device
                    )
                    p.grad.data.add_(noise)

    def _extract_features(self, model, data):
        """提取模型中间层特征用于 MMD 计算"""
        try:
            # 获取底层网络 (PublicLayer/PrivateLayer 包装在 self.layer 中)
            base_model = model.layer if hasattr(model, 'layer') else model
            
            # 针对 CSModels 中 ResNet 的结构:
            #   init_conv -> layers (ModuleList) -> end_layers (AdaptiveAvgPool2d + Flatten + Linear)
            # 我们需要获取 AdaptiveAvgPool2d + Flatten 之后的特征（去掉最后的 Linear）
            if hasattr(base_model, 'init_conv') and hasattr(base_model, 'layers') and hasattr(base_model, 'end_layers'):
                x = base_model.init_conv(data)
                for layer in base_model.layers:
                    x = layer(x)
                # end_layers 是 [AdaptiveAvgPool2d, Flatten, Linear]
                # 只取前两个得到特征向量
                for i, layer in enumerate(base_model.end_layers):
                    if i < len(base_model.end_layers) - 1:  # 跳过最后的 Linear
                        x = layer(x)
                return x  # shape: (batch, 64)

            # MobileNet: features -> avgpool -> classifier
            if hasattr(base_model, 'features') and hasattr(base_model, 'classifier'):
                x = base_model.features(data)
                if hasattr(base_model, 'avgpool'):
                    x = base_model.avgpool(x)
                return x.view(x.size(0), -1)

            # 兜底: 直接使用 logits 作为特征
            with torch.no_grad():
                logits = model(data)
            return logits

        except Exception as e:
            logger.debug(f"[Defence] 特征提取失败: {e}")
            return None

    def _mmd_rbf(self, x, y, sigma=1.0):
        """
        使用 RBF 核计算最大均值差异 (MMD).
        MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        """
        gamma = 1.0 / (2 * sigma ** 2)
        xx = torch.exp(-gamma * torch.cdist(x, x, p=2))
        yy = torch.exp(-gamma * torch.cdist(y, y, p=2))
        xy = torch.exp(-gamma * torch.cdist(x, y, p=2))
        mmd_val = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd_val
