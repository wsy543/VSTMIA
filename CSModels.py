import torch.nn as nn
from normalModel import ResNet, NNModel

class InitParams:
    def __init__(self, args):
        self.args = args
        self.param = self.init_params()

    def init_params(self):
        params = {}
        if (self.args.dataset == 'STL10'):
            params['task'] = 'STL10'
            params['input_size'] = 96
            params['num_classes'] = 10
        elif (self.args.dataset == 'location'):
            params['task'] = 'location'
            params['input_size'] = 446
            params['num_classes'] = 30
        model = self.args.model
        if model == 'resnet':
            params['block_type'] = 'basic'
            params['num_blocks'] = [9, 9, 9]
            params['augment_training'] = True
            params['init_weights'] = True

        elif model == 'nn':
            params['hidden_dim']=[128]
        return params


class PublicLayer(nn.Module):
    def __init__(self, args):
        super(PublicLayer, self).__init__()
        self.params = InitParams(args).init_params()
        self.args = args
        if self.args.model == 'resnet':
            self.layer = ResNet(self.params)
        elif self.args.model == 'nn':
            self.layer=NNModel(self.params)
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
        if self.args.model == 'resnet':
            self.layer = ResNet(self.params)
        elif self.args.model == 'nn':
            self.layer=NNModel(self.params)
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