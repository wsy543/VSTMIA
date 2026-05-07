import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from normalModel import ResNet,MobileNet,VGG,DenseNet,NNModel,ShuffleNetV2,WideResNet,AlexNet,TextCNN
# from groupnormModel import ResNet,MobileNet,VGG,DenseNet

class InitParams:
    def __init__(self, args):
        self.args = args
        self.param = self.init_params()

    def init_params(self):
        params = {}
        if (self.args.dataset == 'CIFAR10'):
            params['task'] = 'cifar10'
            params['input_size'] = 32
            params['num_classes'] = 10
        elif (self.args.dataset == 'CIFAR100'):
            params['task'] = 'cifar100'
            params['input_size'] = 32
            params['num_classes'] = 100
        elif (self.args.dataset == 'STL10'):
            params['task'] = 'STL10'
            params['input_size'] = 96
            params['num_classes'] = 10
        elif (self.args.dataset == 'EuroSAT'):
            params['task'] = 'EuroSAT'
            params['input_size'] = 64
            params['num_classes'] = 10
        elif (self.args.dataset == 'CINIC10'):
            params['task'] = 'cinic10'
            params['input_size'] = 32
            params['num_classes'] = 10
        elif (self.args.dataset == 'GTSRB'):
            params['task'] = 'gtsrb'
            params['input_size'] = 32
            params['num_classes'] = 43
        elif (self.args.dataset == '20Newsgroups'):
            params['task'] = '20Newsgroups'
            params['input_size'] = 500
            params['num_classes'] = 20
        elif (self.args.dataset == 'tinyimagenet'):
            params['task'] = 'tinyimagenet'
            params['input_size'] = 64
            params['num_classes'] = 200
        elif (self.args.dataset == 'texas'):
            params['task'] = 'texas'
            params['input_size'] = 6169
            params['num_classes'] = 100
        elif (self.args.dataset == 'purchase'):
            params['task'] = 'purchase'
            params['input_size'] = 600
            params['num_classes'] = 100
        elif (self.args.dataset == 'location'):
            params['task'] = 'location'
            params['input_size'] = 446
            params['num_classes'] = 30
        elif (self.args.dataset == 'DBP'):
            params['task'] = 'DBP'
            params['input_size'] = 5000
            params['num_classes'] = 70
        elif (self.args.dataset == 'OCT'):
            params['task'] = 'OCT'
            params['input_size'] = 224
            params['num_classes'] = 4
        elif (self.args.dataset == 'imdb'):
            params['task'] = 'imdb'
            params['input_size'] = 1000
            params['num_classes'] = 2
        elif (self.args.dataset == 'yahoo'):
            params['task'] = 'yahoo'
            params['vocab_size'] = 30000
            params['embedding_dim']=100
            params['num_classes'] = 10
            params['sequence_length']=500
        model = self.args.model
        if model == 'cnn':
            print('Using a multilayer convolution neural network based model...')
        elif model == 'vgg':
            # print('Using vgg model...')
            # 继续组装params，字典类型
            params['conv_channels'] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            params['fc_layers'] = [512, 512]
            params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
            params['conv_batch_norm'] = True
            params['init_weights'] = True  # 运行初始化函数initialize_weights，在每个模型中都有。
            params['augment_training'] = True

        elif model == 'resnet':
            # print('Using resnet model...')
            params['block_type'] = 'basic'
            params['num_blocks'] = [9, 9, 9]
            params['augment_training'] = True
            params['init_weights'] = True  # 在 ResNet的初始化函数中，也没有判断这个，直接必选初始化了。所以这里填不填无所谓。

        # elif model == 'wideresnet':
        #     # print('Using wideresnet model...')
        #     params['block_type'] = 'bottle'
        #     params['num_blocks'] = [5, 5, 5]
        #     params['widen_factor'] = 4
        #     params['dropout_rate'] = 0.3
        #     params['augment_training'] = True
        #     params['init_weights'] = True

        elif model == 'mobilenet':
            # print('Using mobilenet model...')
            params['cfg'] = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
            params['augment_training'] = True
        
        elif model == 'densenet':
            params['growth_rate']=32
            params['block_config']=[6,12,24,16]
            params['num_init_features']=64

        elif model == 'shufflenet':
            # shufflenetv2 1.0x output channels
            params['stages_repeats'] = [4, 8, 4]
            params['stages_out_channels'] = [24, 116, 232, 464, 1024]
        elif model == 'wideresnet':
            params['widen_factor'] = 10
            params['dropRate'] = 0
            params['depth'] = 28
        elif model == 'nn':
            params['hidden_dim']=[128]
        elif model == 'alexnet':
            params['dropout']=0.5
        elif model == 'textcnn':
            pass
        return params


class PublicLayer(nn.Module):
    def __init__(self, args):
        super(PublicLayer, self).__init__()
        self.params = InitParams(args).init_params()
        self.args = args
        if self.args.model == 'vgg':
            self.layer= VGG(self.params)
        elif self.args.model == 'resnet':
            self.layer = ResNet(self.params)
        elif self.args.model == 'mobilenet':
            self.layer = MobileNet(self.params)
        elif self.args.model == 'densenet':
            self.layer= DenseNet(self.params)
        elif self.args.model =='shufflenet':
            self.layer = ShuffleNetV2(params=self.params)
        elif self.args.model == 'nn':
            self.layer=NNModel(self.params)
        elif self.args.model == 'wideresnet':
            self.layer = WideResNet(self.params)
        elif self.args.model == 'alexnet':
            self.layer = AlexNet(self.params)
        elif self.args.model=='textcnn':
            self.layer = TextCNN(params=self.params)
        else:
            print('classifier is error')

    def forward(self, x):
        x = self.layer(x)
        return x


class PrivateLayer(nn.Module):
    def __init__(self, args):
        super(PrivateLayer, self).__init__()
        self.params = InitParams(args).init_params()
        self.args = args
        if self.args.model == 'vgg':
            self.layer = VGG(self.params)
        elif self.args.model == 'resnet':
            self.layer = ResNet(self.params)
        elif self.args.model == 'mobilenet':
            self.layer = MobileNet(self.params)
        elif self.args.model == 'densenet':
            self.layer = DenseNet(self.params)
        elif self.args.model =='shufflenet':
            self.layer = ShuffleNetV2(params=self.params)
        elif self.args.model == 'nn':
            self.layer=NNModel(self.params)
        elif self.args.model == 'wideresnet':
            self.layer = WideResNet(self.params)
        elif self.args.model == 'alexnet':
            self.layer = AlexNet(self.params)
        elif self.args.model=='textcnn':
            self.layer = TextCNN(params=self.params)
        else:
            print('classifier is error')
    def forward(self, x):
        x = self.layer(x)
        return x


class ClientModel(nn.Module):
    def __init__(self, args, pub_model, pri_model):
        super(ClientModel, self).__init__()
        self.args = args
        self.feature_extractor = pub_model
        self.classifier = pri_model

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
        max = torch.max(fwd)
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


class VGG_1(nn.Module):
    def __init__(self, params):
        super(VGG_1, self).__init__()
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

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)
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

class VGG_Conv(nn.Module):
    def __init__(self, params):
        super(VGG_Conv, self).__init__()
        self.input_size = int(params['input_size'])
        self.conv_channels = params['conv_channels']
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        
        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        self.cur_input_size = cur_input_size
        self.final_channels = self.conv_channels[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VGG_2(nn.Module):
    def __init__(self, params):
        super(VGG_2, self).__init__()
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels']
        self.fc_layer_sizes = params['fc_layers']
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        self.num_output = 1

        # 创建空的卷积和全连接层
        self.init_conv = nn.Sequential()
        self.layers = nn.ModuleList()

        # 计算 `fc_input_size`，但不实例化卷积层
        self.fc_input_size = self.calculate_fc_input_size()

        # 构建全连接层
        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (self.fc_input_size, width)
            flatten = (layer_id == 0)
            self.layers.append(FcBlock(fc_params, flatten=flatten))
            self.fc_input_size = width

        # 构建最终的输出层
        end_layers = [
            nn.Linear(self.fc_input_size, self.fc_layer_sizes[-1]),
            nn.Dropout(0.5),
            nn.Linear(self.fc_layer_sizes[-1], self.num_classes)
        ]
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def calculate_fc_input_size(self):
        # 使用一个模拟张量来计算经过卷积层后的特征图大小
        x = torch.rand(1, 3, self.input_size, self.input_size)  # 3 为输入通道数（RGB）
        input_channel = 3
        for layer_id, channel in enumerate(self.conv_channels):
            # 卷积层
            x = nn.Conv2d(input_channel, channel, kernel_size=3, padding=1)(x)
            if self.conv_batch_norm:
                x = nn.BatchNorm2d(channel)(x)
            x = nn.ReLU()(x)
            input_channel = channel
            # 池化层
            if self.max_pool_sizes[layer_id] == 2:
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # 展平后的全连接输入大小
        return x.numel()

    def forward(self, x):
        fwd = self.init_conv(x)
        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_FC(nn.Module):
    def __init__(self, conv_output_size, params):
        super(VGG_FC, self).__init__()
        self.fc_layer_sizes = params['fc_layers']
        self.num_classes = int(params['num_classes'])

        fc_input_size = conv_output_size * conv_output_size * params['conv_channels'][-1]
        self.layers = nn.ModuleList()

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = layer_id == 0
            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        # 最后的全连接层
        end_layers = [
            nn.Flatten(),
            nn.Linear(fc_input_size, self.fc_layer_sizes[-1]),
            nn.Dropout(0.5),
            nn.Linear(self.fc_layer_sizes[-1], self.num_classes)
        ]
        self.end_layers = nn.Sequential(*end_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.end_layers(x)
        return x
    
def initialize_vgg_models(params):
    # 初始化卷积层部分
    vgg_conv = VGG_Conv(params)
    # 计算卷积输出的尺寸
    conv_output_size = vgg_conv.cur_input_size

    # 初始化全连接层部分
    vgg_fc = VGG_FC(conv_output_size, params)

    return vgg_conv, vgg_fc

def initialize_densenet_models(params):
    # 初始化卷积层部分
    densenet_conv = DenseNet_1(params)
    # 计算卷积输出的尺寸
    conv_output_size = densenet_conv.num_features
    densenet_FC = DenseNet_2(params,conv_output_size)

    return densenet_conv, densenet_FC


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        
        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=False))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=False))

        self.layers = layers
            
    def forward(self, x):
        fwd = self.layers[0](x) 
        fwd += self.layers[1](x) 
        fwd = self.layers[2](fwd) 
        return fwd

class ResNet_1(nn.Module):
    def __init__(self, params):
        super(ResNet_1, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']

        self.in_channels = 16
        self.num_output =  1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=False))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))
        
        self.initialize_weights()

        self.augment_training = params['augment_training']


    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.init_conv(x)
        for layer in self.layers:
            out = layer(out)
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

class ResNet_2(nn.Module):
    def __init__(self, params):
        super(ResNet_2, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']

        self.in_channels = 16
        self.num_output =  1

        if self.block_type == 'basic':
            self.block = BasicBlock
        
        end_layers = []

        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

        self.augment_training = params['augment_training']

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.end_layers(x)
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

class MobileNet_1(nn.Module):
    def __init__(self, params):
        super(MobileNet_1, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])

        self.num_output = 1
        self.in_channels = 32
        init_conv = []
        
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=False))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))


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
        return fwd
    
class MobileNet_2(nn.Module):
    def __init__(self, params):
        super(MobileNet_2, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])

        self.num_output = 1
        self.in_channels = 32
    
        end_layers = []

        end_layers.append(nn.AvgPool2d(2))

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
        fwd = self.end_layers(x)
        return fwd

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.GroupNorm(in_channels)
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
        # self.bn = nn.BatchNorm2d(in_channels)
        self.bn = nn.GroupNorm(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x
    

class DenseNet_1(nn.Module):
    def __init__(self, params):
        super(DenseNet_1, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, params['num_init_features'], kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(params['num_init_features'])
        self.bn1 = nn.GroupNorm(params['num_init_features'])
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建 Dense Block 和 Transition Layer
        num_features = params['num_init_features']
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(params['block_config']):
            # Dense Block
            dense_block = DenseBlock(num_layers, num_features, params['growth_rate'])
            self.dense_blocks.append(dense_block)
            num_features += num_layers * params['growth_rate']
            # Transition Layer
            if i != len(params['block_config']) - 1:
                transition_layer = TransitionLayer(num_features, num_features // 2)
                self.transition_layers.append(transition_layer)
                num_features = num_features // 2

        self.num_features = num_features

        # 最后的 Batch Norm
        # self.bn2 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.GroupNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        return x
    

class DenseNet_2(nn.Module):
    def __init__(self, params,num_features):
        super(DenseNet_2, self).__init__() 
        self.fc = nn.Linear(num_features, params['num_classes'])

    def forward(self, x):
        # 最终全连接层
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x