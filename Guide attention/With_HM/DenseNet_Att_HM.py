import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition, model_urls

class AttentionNet(nn.Module):
    def __init__(self, channels, hidden_layers=1, kernel_att=3, stride_att=1):
        super(AttentionNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.kernel_att = kernel_att
        self.stride_att = stride_att
        self.padding_att = (self.kernel_att - 1) // 2

        self.c1 = nn.Conv2d(channels, channels, kernel_size=self.kernel_att, stride=self.stride_att,
                            padding=self.padding_att)
        self.relu = nn.ReLU()
        if self.hidden_layers >= 2:
            self.c2 = nn.Conv2d(channels, channels, kernel_size=self.kernel_att, stride=self.stride_att,
                                padding=self.padding_att)
        if self.hidden_layers >= 3:
            self.c3 = nn.Conv2d(channels, channels, kernel_size=self.kernel_att, stride=self.stride_att,
                                padding=self.padding_att)
        if self.hidden_layers >= 4:
            self.c4 = nn.Conv2d(channels, channels, kernel_size=self.kernel_att, stride=self.stride_att,
                                padding=self.padding_att)
        self.c5 = nn.Conv2d(channels, channels, kernel_size=self.kernel_att, stride=self.stride_att,
                            padding=self.padding_att)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        if self.hidden_layers >= 2:
            x = self.c2(x)
            x = self.relu(x)
        if self.hidden_layers >= 3:
            x = self.c3(x)
            x = self.relu(x)
        if self.hidden_layers >= 4:
            x = self.c4(x)
            x = self.relu(x)
        x = self.c5(x)


        return self.sig(x)

class DenseNet_multi_att_HM(nn.Module):
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
                 bp_position=False, all_in_1_reduction=False, hidden_layers_att=1, kernel_att=3, stride_att=1):

        super(DenseNet_multi_att_HM, self).__init__()
        #self.gradients = None

        self.bp_position = bp_position
        self.all_in_1_reduction = all_in_1_reduction
        self.hidden_layers_att = hidden_layers_att
        self.kernel_att = kernel_att
        self.stride_att = stride_att
        self.padding_att = (self.kernel_att - 1) // 2


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
        self.conv_reducer_1 = nn.Conv2d(1664, 64, 1)
        self.con_relu_1 = nn.ReLU()
        self.conv_reducer_2 = nn.Conv2d(64, 32, 1)
        self.con_relu_2 = nn.ReLU()
        self.conv_reducer_3 = nn.Conv2d(32, 16, 1)
        self.con_relu_3 = nn.ReLU()
        self.conv_reducer_4 = nn.Conv2d(16, 8, 1)
        self.con_relu_4 = nn.ReLU()

        self.conv_reducer_all_in_1 = nn.Conv2d(1664, 8, 1)

        #Guide attention
        self.multu_attention = AttentionNet(1664, self.hidden_layers_att, self.kernel_att, self.stride_att)


        # Linear layer for localization
        self.classifier_locations = nn.Linear(2048, 7)

        # Official init from torch repo.

        # Linear layer for radiographic finding
        self.classifier = nn.Linear(num_features, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, no_grad=False):
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
        if (no_grad is False) and (self.training is False):
            print('Gradients Hooked')
            h = out.register_hook(self.activations_hook)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        #attention_layer = F.adaptive_avg_pool2d(attention_layer, (1, 1)).view(features.size(0), -1)
        radiographical_findings = self.classifier(out)

        return radiographical_findings, locations

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
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
        # print(out_pos.shape)
        # print(attention_layer.shape)

        # attention
        # print(attention_layer.shape)

        out = out * attention_layer
        return out


def densenet_att_hm_169(pretrained=False, all_in_1_reduction=False, bp_elementwise=True, hidden_layers_att=1,
                        kernel_att=3, stride_att=1, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_multi_att_HM(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), all_in_1_reduction=all_in_1_reduction, bp_position=bp_elementwise,
                                  hidden_layers_att=hidden_layers_att, kernel_att=kernel_att, stride_att=stride_att, **kwargs)
    mModel_dict = model.state_dict()

    #model.multu_attention = AttentionNet(16, hidden_layers_att)

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



        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        model.classifier_locations.in_features = 2048
        mModel_dict.update(state_dict)
        for k in mModel_dict.keys():
            print(k)
        model.load_state_dict(mModel_dict)


    return model


def loadSD_densenet_att_hm_169(pretrained=True, bp_elementwise=True, hidden_layers_att=1, model_state_dict='',
                               kernel_att=3, stride_att=1, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet_multi_att_HM(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), hidden_layers_att=hidden_layers_att,
                                  bp_position=bp_elementwise, kernel_att=kernel_att, stride_att=stride_att, **kwargs)

    #model.multu_attention = AttentionNet(16, hidden_layers_att)
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    #mModel_dict = model.state_dict()

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_state_dict
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        #mModel_dict.update(state_dict)
        model.load_state_dict(state_dict)
        #model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()

    return model