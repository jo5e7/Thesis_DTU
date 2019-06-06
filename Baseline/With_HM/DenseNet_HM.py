import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.models import densenet169
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition, model_urls

class DenseNet_MH(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 all_in_1_reduction=False):

        super(DenseNet_MH, self).__init__()

        self.all_in_1_reduction = all_in_1_reduction

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        # get the classifier of the densenet
        self.classifier = nn.Linear(num_features, 1)

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, no_grad=False):
        x = self.features(x)
        # register the hook
        if (no_grad is False) and (self.training is False):
            print('Gradients Hooked')
            h = x.register_hook(self.activations_hook)
        x = F.relu(x)




        # don't forget the pooling
        #print(x)
        #print(features.size(0))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)

def densenet_hm_169(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_MH(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    mModel_dict = model.state_dict()

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            #print(key)
            #print(res)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        mModel_dict.update(state_dict)
        model.load_state_dict(mModel_dict)

    return model

def loadSD_densenet_hm_169(pretrained=True, model_url='', **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_MH(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    #mModel_dict = model.state_dict()

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_url
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        #mModel_dict.update(state_dict)
        #model.load_state_dict(mModel_dict)
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    return model