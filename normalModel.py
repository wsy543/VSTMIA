import torch.nn as nn
import torch


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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class NNModel(nn.Module):
    def __init__(self, params):
        super(NNModel, self).__init__()

        self.input_dim = params['input_size']
        self.hidden_dims = params['hidden_dim']
        self.output_dim = params['num_classes']

        self.layers = nn.ModuleList()
        current_dim = self.input_dim

        for h_dim in self.hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            
            self.layers.append(nn.LayerNorm(h_dim))
            
            self.layers.append(nn.ReLU())
            
            self.layers.append(nn.Dropout(0.2)) 
            
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim, self.output_dim)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = layer(x)

        x = self.output_layer(x)

        return x