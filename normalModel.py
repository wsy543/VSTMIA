import torch.nn as nn
import torch
import math
from typing import List, Callable
from torch import Tensor
import torch.nn.functional as F
import math


class VGG(nn.Module):
    def __init__(self, params):
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels']
        self.fc_layer_sizes = params['fc_layers']

        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        # if 'distill' in args.mode:
        #    self.train_func = utils.cnn_train_dis
        # else:
        #    self.train_func = utils.cnn_train
        # self.test_func = utils.cnn_test
        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']

        self.in_channels = 16
        self.num_output = 1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.AdaptiveAvgPool2d((1,1)))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64 * self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

        self.augment_training = params['augment_training']

        # if 'distill' in args.mode:
        #    self.train_func = utils.cnn_train_dis
        # else:
        #    self.train_func = utils.cnn_train
        # self.test_func = utils.cnn_test

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x).clone()
        fwd = fwd + self.layers[1](x).clone()
        fwd = self.layers[2](fwd).clone()
        return fwd


class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = torch.cat([x, out], 1)  # 拼接输入和输出
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        # 初始卷积层
        self.num_classes=params['num_classes']
        self.growth_rate=params['growth_rate']
        self.block_config = params['block_config']
        self.num_init_features=params['num_init_features']
        self.conv1 = nn.Conv2d(3, self.num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建 Dense Block 和 Transition Layer
        num_features = self.num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(self.block_config):
            # Dense Block
            dense_block = DenseBlock(num_layers, num_features, self.growth_rate)
            self.dense_blocks.append(dense_block)
            num_features += num_layers * self.growth_rate
            # Transition Layer
            if i != len(self.block_config) - 1:
                transition_layer = TransitionLayer(num_features, num_features // 2)
                self.transition_layers.append(transition_layer)
                num_features = num_features // 2

        # 最后的 Batch Norm
        self.bn2 = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        # 初始卷积层
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Dense Block 和 Transition Layer
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)

        # 最终全连接层
        x = self.bn2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class MobileNet(nn.Module):
    def __init__(self, params):
        super(MobileNet, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])

        self.num_output = 1
        self.in_channels = 32
        init_conv = []
        
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []

        # end_layers.append(nn.AvgPool2d(2))
        end_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        end_layers.append(Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd
    

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,params,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        self.stages_repeats = params['stages_repeats']
        self.stages_out_channels = params['stages_out_channels']
        self.num_classes=params['num_classes']

        if len(self.stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(self.stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = self.stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, self.stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, self.num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



# class NNModel(nn.Module):
#     def __init__(self, params):
#         super(NNModel, self).__init__()

#         self.input_dim = params['input_size']
#         # hidden_dim 现在是一个列表，例如 [64, 32, 16]
#         self.hidden_dims = params['hidden_dim']
#         self.output_dim = params['num_classes']

#         # 使用 ModuleList 来存放动态创建的层和激活函数
#         # ModuleList 是一个特殊类型的 list，它可以正确注册其包含的 Module，
#         # 使得模型知道其内部有哪些子模块，这对于保存/加载模型参数等操作很重要。
#         self.layers = nn.ModuleList()

#         # 当前层的输入维度，最开始是模型的总输入维度
#         current_dim = self.input_dim

#         # 动态创建隐藏层和 ReLU 激活函数
#         # 遍历 hidden_dims 列表中的每个维度
#         for h_dim in self.hidden_dims:
#             # 添加一个全连接层，输入维度是 current_dim，输出维度是 h_dim
#             self.layers.append(nn.Linear(current_dim, h_dim))
#             # 在每个隐藏层之后添加一个 ReLU 激活函数
#             self.layers.append(nn.ReLU())
#             # 当前层的输出维度成为下一层的输入维度
#             current_dim = h_dim

#         # 添加最终的输出层
#         # 输出层的输入维度是最后一个隐藏层的输出维度 (即循环结束后的 current_dim)
#         # 输出维度是模型的总输出类别数
#         self.layers.append(nn.Linear(current_dim, self.output_dim))

#         # 定义 Softmax 激活函数，用于将输出转换为概率分布
#         self.softmax = nn.Softmax(dim=1) # Softmax 通常作用于类别维度 (dim=1)

#     def forward(self, x):
#         # 按照 ModuleList 中的顺序，依次通过每一层
#         # 这就实现了根据 self.layers 中的层数进行动态前向传播
#         for layer in self.layers:
#             x = layer(x)

#         # 在所有线性层和 ReLU 之后，应用 Softmax
#         x = self.softmax(x)

#         return x

class NNModel(nn.Module):
    def __init__(self, params):
        super(NNModel, self).__init__()

        self.input_dim = params['input_size']  # 应该是 600
        self.hidden_dims = params['hidden_dim'] # 例如 [512, 256, 128]
        self.output_dim = params['num_classes'] # 100

        self.layers = nn.ModuleList()
        current_dim = self.input_dim

        # === 动态构建隐藏层 ===
        for h_dim in self.hidden_dims:
            # 1. 线性层
            self.layers.append(nn.Linear(current_dim, h_dim))
            
            # 2. [关键修改] 加入 LayerNorm
            #这能解决 Non-IID 下梯度不稳定的问题
            self.layers.append(nn.LayerNorm(h_dim))
            
            # 3. 激活函数
            self.layers.append(nn.ReLU())
            
            # 4. [关键修改] 加入 Dropout
            # Purchase100 很容易过拟合，Dropout 必不可少
            self.layers.append(nn.Dropout(0.2)) 
            
            current_dim = h_dim

        # === 最终输出层 ===
        self.output_layer = nn.Linear(current_dim, self.output_dim)

    def forward(self, x):
        # 确保输入是 float32 (防止 Double/Float 类型不匹配报错)
        if x.dtype != torch.float32:
            x = x.float()
        
        # 展平输入 (处理可能的 (Batch, 1, 600) 情况)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # 通过所有隐藏层
        for layer in self.layers:
            x = layer(x)

        # 通过输出层
        x = self.output_layer(x)

        # [重要] 这里不要加 Softmax！
        # nn.CrossEntropyLoss 会自动帮你做 Softmax
        return x

class BasicBlock_wide(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock_wide, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, params):
        super(WideResNet, self).__init__()
        widen_factor = params['widen_factor']
        dropRate = params['dropRate']
        num_classes = params['num_classes']
        depth = params['depth']
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock_wide
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



class AlexNet(nn.Module):
    def __init__(self, params):
        super(AlexNet, self).__init__()
        num_classes=params['num_classes']
        dropout = params['dropout']
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 修改卷积核大小和步幅
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 使用自适应平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, params, kernel_sizes=[3, 4, 5], num_filters=100, dropout_prob=0.5):
        super(TextCNN, self).__init__()
        # Embedding Layer
        # input_dim: 词汇表大小
        # output_dim: 词向量维度
        num_classes = params['num_classes']
        vocab_size = params['vocab_size']
        embedding_dim = params['embedding_dim']
        sequence_length = params['sequence_length']


        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Conv1D layers for different kernel sizes (示例中使用多个 kernel size)
        # 您提供的 Keras 模型是多个 Conv1D 接 MaxPool，这里实现一个更经典的 TextCNN 结构，
        # 使用不同大小的卷积核并行提取特征，然后池化。
        # 如果您严格需要 Keras 图中的层序结构，我可以修改。
        # 按照Keras图的层序结构，Conv1D后面直接接Conv1D和MaxPool，filters数量递增
        # 这里按照Keras图的逻辑来构建层
        self.conv1a = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=5, padding='same')
        self.conv1b = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25) # Keras图是0.25

        self.conv2a = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same')
        self.conv2b = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3a = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.conv3b = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)

        # Flattened size calculation
        # sequence_length -> sequence_length/2 -> sequence_length/4 -> sequence_length/8
        # final sequence length after pooling = sequence_length // 8
        # final channels = 256
        flattened_features = (sequence_length // 8) * 256

        # Dense layers
        self.fc1 = nn.Linear(flattened_features, 256) # 根据Keras图是256
        self.fc2 = nn.Linear(256, num_classes) # 输出层

        self.dropout_fc = nn.Dropout(0.5) # Keras图flatten后面有一个Dropout

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length) - 整数 token ID 序列

        # Embedding layer
        # output shape: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)

        # PyTorch Conv1d 需要输入形状为 (batch_size, channels, sequence_length)
        # 所以需要对 Embedding 的输出进行转置
        x = x.permute(0, 2, 1) # output shape: (batch_size, embedding_dim, sequence_length)

        # Conv1D and Pooling Blocks
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1) # output shape: (batch_size, flattened_features)

        # Dense layers
        x = self.dropout_fc(x) # Keras图flatten后面有一个Dropout
        x = F.relu(self.fc1(x))
        out = self.fc2(x) # 输出层通常不加 softmax, 由损失函数处理

        return out