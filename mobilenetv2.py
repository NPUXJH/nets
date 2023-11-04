import torch
import torch.nn as nn

#------------------------------------------------------------------------------------#
# mobilenet v2的结构特点是引入了非传统残差网络的逆残差结构
# 传统的残差连接的残差边是连接了两个通道较深的特征层
# mobilenet v2的残差边连接的则是两个通道较浅的特征层
#------------------------------------------------------------------------------------#

# 定义卷积块,conv+bn+relu6
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        # featuremap尺寸严格除以2，故用此计算padding的值
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

#-----------------------------------------------------------------------------------------------#
# 此处的expand_ratio指的是网络原文中的参数k，表示block在实现中间过程中，第一个1*1卷积层扩展到输入通道数的倍率
# stride是中间的3*3卷积层的步长
# -----------------------------------------------------------------------------------------------#
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channels = in_channels * expand_ratio
        # 当输入输出通道数相等，且图像尺寸相同时，采用残差连接，这里是一个信号，用于前向传播函数的判断
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        # 如果expand_ratio不为1，则说明此时需要进行1*1卷积进行pointwise升维
        if expand_ratio != 1:
            layers += (ConvBNReLU(in_channels, hidden_channels, kernel_size=1, stride=1))
        layers += [
            # 添加conv+bn+relu6，此处使用组数等于输入通道数的分组卷积实现depthwise conv
            ConvBNReLU(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 此处比在进行ReLU6激活，而是用线性激活，及激活函数为空
        ]

        self.inverted_residualblock = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.inverted_residualblock(x)
        else:
            return self.inverted_residualblock(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.classes = num_classes
        self.feature = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2),    # conv+bn+relu6,(n,3,224,224)-->(n,32,112,112)
            InvertedResidual(32, 16, 1, 1), # inverted residual block,(n,32,112,112)-->(n,16,112,112)
            InvertedResidual(16, 24, 2, 6), # inverted residual block,(n,16,112,112)-->(n,24,56,56)
            InvertedResidual(24, 24, 1, 6), # inverted residual block,(n,24,56,56)-->(n,24,56,56)
            InvertedResidual(24, 32, 2, 6), # inverted residual block,(n,24,56,56)-->(n,24,56,56)
            InvertedResidual(32, 32, 1, 6), # inverted residual block,(n,32,28,28)-->(n,32,28,28)
            InvertedResidual(32, 32, 1, 6), # inverted residual block,(n,32,28,28)-->(n,32,28,28)
            InvertedResidual(32, 64, 2, 6), # inverted residual block,(n,32,28,28)-->(n,64,14,14)
            InvertedResidual(64, 64, 1, 6), # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 64, 1, 6), # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 64, 1, 6), # inverted residual block,(n,64,14,14)-->(n,64,14,14)
            InvertedResidual(64, 96, 1, 6), # inverted residual block,(n,64,14,14)-->(n,96,14,14)
            InvertedResidual(96, 96, 1, 6), # inverted residual block,(n,96,14,14)-->(n,96,14,14)
            InvertedResidual(96, 96, 1, 6), # inverted residual block,(n,96,14,14)-->(n,96,14,14)
            InvertedResidual(96, 160, 2, 6),    # inverted residual block,(n,96,14,14)-->(n,160,7,7)
            InvertedResidual(160, 160, 1, 6),   # inverted residual block,(n,160,7,7)-->(n,160,7,7)
            InvertedResidual(160, 160, 1, 6),   # inverted residual block,(n,160,7,7)-->(n,160,7,7)
            InvertedResidual(160, 320, 1, 6),   # inverted residual block,(n,160,7,7)-->(n,320,7,7)
            ConvBNReLU(320, 1280, 1,1)  # conv+bn+relu6,(n,320,7,7)-->(n,1280,7,7)
        )

        #------------------------------------------------------------#
        # 这里是分类头的定义
        # 采用全局平均池化来取代全连接层，有效地减少了参数量
        #------------------------------------------------------------#
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # avgpool,(n,1280,7,7)-->(n,1280,1,1)
            nn.Conv2d(1280, self.classes, 1, 1, 0)  # 1x1conv,(n,1280,1,1)-->(n,num_classes,1,1),等同于linear
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        # 压缩不需要的维度，返回分类结果,(n,num_classes,1,1)-->(n,num_classes)
        return x.view(-1, self.classes)

if __name__ == '__main__':
    # ‘DenseNet‘, ‘densenet121‘, ‘densenet169‘, ‘densenet201‘, ‘densenet161‘
    # Example
    net = MobileNetV2(1000)
    print(net)