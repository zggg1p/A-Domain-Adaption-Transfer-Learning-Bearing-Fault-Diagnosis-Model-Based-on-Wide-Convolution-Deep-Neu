import torch.nn as nn
import torch


from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # pw1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # dw1
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # pw2
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class MobileNetV2(nn.Module):

    def __init__(self, block, class_num):
        # block is the basic module which should be leaded in
        self.inplanes = 32
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, 1, stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 32, 1, stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 64, 1, stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 96, 2, stride=2, expansion=6)
        self.conv5 = nn.Conv2d(96, 128, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output=50*128*1*1
        self.conv7 = nn.Conv2d(128, class_num, kernel_size=1, stride=1, bias=False)  # fault clssses


        self.domain_classifier = nn.Sequential(

            nn.Linear(in_features=128, out_features=500),
            nn.LeakyReLU(),
            nn.Linear(in_features=500, out_features=500),
            nn.LeakyReLU(),
            nn.Linear(in_features=500, out_features=2)
        )

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x1 = self.bn1(x)
    #     x2 = self.relu(x1)
    #     x3 = self.layer1(x2)
    #     x4 = self.layer2(x3)
    #     x5 = self.layer3(x4)
    #     x6 = self.layer4(x5)
    #     x7 = self.conv5(x6)
    #     x8 = self.avgpool(x7)
    #     x9 = self.conv7(x8)
    #     x9 = x9.view(x9.size(0), -1)
    #     output = F.log_softmax(x9, dim=1)
    #
    #     return output

    def forward(self, x, alpha):
        x = self.conv1(x)
        x1 = self.bn1(x)
        x2 = self.relu(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x7 = self.conv5(x6)
        x8 = self.avgpool(x7)
        x9 = self.conv7(x8)
        x9 = x9.view(x9.size(0), -1)
        output = F.log_softmax(x9, dim=1)

        reversed_feature = ReverseLayerF.apply(x8.view(x8.size(0), -1), alpha)
        x_10 = self.domain_classifier(reversed_feature)
        domain_out = F.log_softmax(x_10, dim=1)

        return output, domain_out


class DANN(nn.Module):

    def __init__(self, class_num):
        super(DANN, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=5, stride=2))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5, stride=2))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.fc = nn.Sequential()
        self.fc.add_module('c_fc1', nn.Linear(50 * 13 * 13, 100))
        self.fc.add_module('c_relu1', nn.ReLU(True))
        self.fc.add_module('c_fc2', nn.Linear(100, 100))
        self.fc.add_module('c_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_classifier', nn.Linear(100, class_num))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(in_features=100, out_features=100))
        self.domain_classifier.add_module('d-l-relu', nn.LeakyReLU())
        self.domain_classifier.add_module('d_classifier', nn.Linear(in_features=100, out_features=2))
        self.domain_classifier.add_module('d-softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        class_output = self.class_classifier(x)

        reversed_feature = ReverseLayerF.apply(x, alpha)
        domain_out = self.domain_classifier(reversed_feature)

        return class_output, domain_out


class WDCNN(nn.Module):
    def __init__(self, C_in, class_num):
        super(WDCNN, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output



class SiameseNet(nn.Module):
    def __init__(self, C_in, class_num):
        super(SiameseNet, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.classifier(output)
        return output



class TINet(nn.Module):

    def __init__(self, C_in, class_num):
        super(TINet, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(C_in, 16, 64, stride=8, padding=27, bias=False),
            nn.Dropout(0.3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 100),
            nn.Linear(100, class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

if __name__ == '__main__':
    print(MobileNetV2(Bottleneck, 10))
    # print(DANN(10))



class WDCNN1(nn.Module):
    def __init__(self, C_in, class_num):
        super(WDCNN1, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(in_features=100, out_features=100))
        self.domain_classifier.add_module('d-l-relu', nn.LeakyReLU())
        self.domain_classifier.add_module('d_classifier', nn.Linear(in_features=100, out_features=2))
        self.domain_classifier.add_module('d-softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha):
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        class_output = self.classifier(x)

        reversed_feature = ReverseLayerF.apply(x, alpha)
        domain_out = self.domain_classifier(reversed_feature)
        return class_output, domain_out


class WDCNN2(nn.Module):
    def __init__(self, C_in, class_num):
        super(WDCNN2, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.Softmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, alpha):
        x = self.net(x)
        output1 = x.view(x.size()[0], -1)
        x = x.view(x.size()[0], -1)
        output2 = x.view(x.size()[0], -1)
        x = self.fc(x)
        output3 = x.view(x.size()[0], -1)
        class_output = self.classifier(x)

        reversed_feature = ReverseLayerF.apply(x, alpha)
        domain_out = self.domain_classifier(reversed_feature)
        return output1, output2, output3, class_output, domain_out



class WDCNN3(nn.Module):
    def __init__(self, C_in, class_num):
        super(WDCNN3, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(in_features=100, out_features=100))
        self.domain_classifier.add_module('d-l-relu', nn.LeakyReLU())
        self.domain_classifier.add_module('d_classifier', nn.Linear(in_features=100, out_features=2))
        self.domain_classifier.add_module('d-softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha):
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        class_output = self.classifier(x)

        reversed_feature = ReverseLayerF.apply(x, alpha)
        domain_out = self.domain_classifier(reversed_feature)
        return class_output, domain_out


class WDCNN4(nn.Module):
    def __init__(self, C_in, class_num):
        super(WDCNN4, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Dense layer  全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=class_num),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = self.net(x)
        output1 = x.view(x.size()[0], -1)
        x = x.view(x.size()[0], -1)
        output2 = x.view(x.size()[0], -1)
        x = self.fc(x)
        output3 = x.view(x.size()[0], -1)
        output = self.classifier(x)

        return output1, output2, output3, output
