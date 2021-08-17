import torch
import torch.nn as nn
import os
from div.download_from_url import download_from_url

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pretrained')

__all__ = ['iResNet', 'iresnet18', 'iresnet34', 'iresnet50', 'iresnet101',
           'iresnet152', 'iresnet200', 'iresnet302', 'iresnet404', 'iresnet1001']

model_urls = {
    'iresnet18': 'Trained model not available yet!!',
    'iresnet34': 'Trained model not available yet!!',
    'iresnet50': 'https://drive.google.com/uc?export=download&id=1Waw3ob8KPXCY9iCLdAD6RUA0nvVguc6K',
    'iresnet101': 'https://drive.google.com/uc?export=download&id=1cZ4XhwZfUOm_o0WZvenknHIqgeqkY34y',
    'iresnet152': 'https://drive.google.com/uc?export=download&id=10heFLYX7VNlaSrDy4SZbdOOV9xwzwyli',
    'iresnet200': 'https://drive.google.com/uc?export=download&id=1Ao-f--jNU7MYPqSW8UMonXVrq3mkLRpW',
    'iresnet302': 'https://drive.google.com/uc?export=download&id=1UcyvLhLzORJZBUQDNJdsx3USCloXZT6V',
    'iresnet404': 'https://drive.google.com/uc?export=download&id=1hEOHErsD6AF1b3qQi56mgxvYDneTvMIq',
    'iresnet1001': 'Trained model not available yet!!',
}



# 定义residual
class RB(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1):
        super(RB, self).__init__()
        self.rb = nn.Sequential(nn.Conv2d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm2d(nout),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm2d(nout))
    def forward(self, input):
        x = input
        x = self.rb(x)
        return nn.ReLU(input + x)

# 定义SE模块
class SE(nn.Module):
    def __init__(self, nin, nout, reduce=16):
        super(SE, self).__init__()
        self.gp = nn.AvgPool2d(1)
        self.rb1 = RB(nin, nout)
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
    def forward(self, input):
        x = input
        x = self.rb1(x)

        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        if start_block:
            self.bn2 = norm_layer(planes)

        if end_block:
            self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)

        if start_block:
            self.bn3 = norm_layer(planes * self.expansion)

        if end_block:
            self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out

#     CBAM注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        out = max_out
        return self.sigmoid(out)

class iResNet(nn.Module):

    def __init__(self, block, layers, num_classes=500, zero_init_residual=True, norm_layer=None, dropout_prob0=0.0):
        super(iResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        # 网络的卷积层的最后一层加入注意力机制

        self.ca1 = ChannelAttention(self.inplanes)
        # self.sa1 = SpatialAttention()



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))




        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,
                            start_block=True))
        self.inplanes = planes * block.expansion
        exclude_bn0 = True
        for _ in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(block(self.inplanes, planes, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 添加注意力机制
        x = self.ca(x) * x
        # x = self.sa(x) * x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # 添加注意力机制
        x = self.ca1(x) * x
        # x = self.sa1(x) * x


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)

        return x


def iresnet18(pretrained=False, **kwargs):
    """Constructs a iResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet18'],
                                                           root=default_cache_path)))
    return model


def iresnet34(pretrained=False, **kwargs):
    """Constructs a iResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet34'],
                                                           root=default_cache_path)))
    return model


def iresnet50(pretrained=True, **kwargs):
    """Constructs a iResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = iResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet50'],
                                                           root=default_cache_path)))
    return model


def iresnet101(pretrained=False, **kwargs):
    """Constructs a iResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet101'],
                                                           root=default_cache_path)))
    return model


def iresnet152(pretrained=False, **kwargs):
    """Constructs a iResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet152'],
                                                           root=default_cache_path)))
    return model


def iresnet200(pretrained=False, **kwargs):
    """Constructs a iResNet-200 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet200'],
                                                           root=default_cache_path)))
    return model


def iresnet302(pretrained=False, **kwargs):
    """Constructs a iResNet-302 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [4,  34, 58, 4], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet302'],
                                                           root=default_cache_path)))
    return model


def iresnet404(pretrained=False, **kwargs):
    """Constructs a iResNet-404 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [4,  46, 80, 4], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet404'],
                                                           root=default_cache_path)))
    return model


def iresnet1001(pretrained=False, **kwargs):
    """Constructs a iResNet-1001 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResNet(Bottleneck, [4,  155, 170, 4], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresnet1001'],
                                                           root=default_cache_path)))
    return model
