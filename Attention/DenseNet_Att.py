import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition, model_urls

class AttentionNet(nn.Module):
    def __init__(self, size):
        super(AttentionNet, self).__init__()
        self.l1 = nn.Linear(size, size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(size, size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)

        return self.sig(x)





class DenseNet_att(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet_att, self).__init__()


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

        #Attention
        self.attention = AttentionNet(1664)
        # Linear layer for localization
        self.classifier_locations = nn.Linear(num_features, 2)

        # Official init from torch repo.

        # Linear layer for radiographic finding
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        # attention
        attention_input = out.detach()
        attention_layer = self.attention(attention_input)

        out_aux = out.detach()
        out = out * attention_layer
        attention_layer = attention_layer * out_aux

        radiographical_findings = self.classifier(out)
        locations = self.classifier_locations(attention_layer)
        return radiographical_findings, locations

class DenseNet_multi_att(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 bp_position=False, all_in_1_reduction=False):

        super(DenseNet_multi_att, self).__init__()

        self.bp_position = bp_position
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

        # Convolution to reduce dimesion
        self.conv_reducer_1 = nn.Conv2d(1664, 900, 1)
        self.con_relu_1 = nn.ReLU()
        self.conv_reducer_2 = nn.Conv2d(900, 400, 1)
        self.con_relu_2 = nn.ReLU()
        self.conv_reducer_3 = nn.Conv2d(400, 150, 1)
        self.con_relu_3 = nn.ReLU()
        self.conv_reducer_4 = nn.Conv2d(150, 64, 1)
        self.con_relu_4 = nn.ReLU()

        self.conv_reducer_all_in_1 = nn.Conv2d(1664, 64, 1)

        #Attention
        self.multu_attention = AttentionNet(16)


        # Linear layer for localization
        self.classifier_locations = nn.Linear(num_features, 2)

        # Official init from torch repo.

        # Linear layer for radiographic finding
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        # Embeddings from features to positions
        if self.bp_position:
            attention_layer = self.multu_attention(out)
        else:
            attention_input = out.detach()
            attention_layer = self.multu_attention(attention_input)

        # Dimentional reduction
        if self.all_in_1_reduction:
            out_pos = self.conv_reducer_all_in_1(attention_layer)
            out_pos = self.con_relu_1(out_pos)
            pass
        else:
            out_pos = self.conv_reducer_1(attention_layer)
            out_pos = self.con_relu_1(out_pos)
            out_pos = self.conv_reducer_2(out_pos)
            out_pos = self.con_relu_2(out_pos)
            out_pos = self.conv_reducer_3(out_pos)
            out_pos = self.con_relu_3(out_pos)
            out_pos = self.conv_reducer_4(out_pos)
            out_pos = self.con_relu_4(out_pos)
        out_pos = out_pos.view(out_pos.shape[0], -1)
        locations = self.classifier_locations(out_pos)
        #print(out_pos.shape)
        #print(attention_layer.shape)

        # attention
        # print(attention_layer.shape)

        out = out * attention_layer
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        #attention_layer = F.adaptive_avg_pool2d(attention_layer, (1, 1)).view(features.size(0), -1)
        radiographical_findings = self.classifier(out)

        return radiographical_findings, locations


def densenet_att_121(pretrained=False, bp_elementwise=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_att(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bp_elementwise=bp_elementwise,
                     **kwargs)
    model.attention = AttentionNet(1024)
    mModel_dict = model.state_dict()

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        mModel_dict.update(state_dict)
        model.load_state_dict(mModel_dict)

    return model


def densenet_att_169(pretrained=False, bp_elementwise=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_att(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bp_elementwise=bp_elementwise,
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
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        mModel_dict.update(state_dict)
        model.load_state_dict(mModel_dict)

    return model


def densenet_multi_att_169(pretrained=False, bp_position=False, all_in_1_reduction=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_multi_att(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bp_position=bp_position,
                               all_in_1_reduction=all_in_1_reduction,
                     **kwargs)
    model.multu_attention = AttentionNet(16)
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
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        mModel_dict.update(state_dict)
        model.load_state_dict(mModel_dict)
        model.classifier_locations.in_features = 16384

    return model