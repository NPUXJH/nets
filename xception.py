import torch
import torch.nn as nn
import torchvision

#------------------------------------------------------------------------------------------------#
# xcetion网络架构由深度学习框架keras的作者Francois Chollet提出
# ineption系列自2014年GoogleNet（inception v1）问世以来，inception系列持续发展
# 随后出现了inception-BN,inception v2,inception-v3,inception-v4,inception-resnet,xception等网络结构
# 后续的inception v3和xception等网络被广泛用于迁移学习的基网络
#------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------#
# ！！！！！！！！！！！！！！！！！！！！Notice！！！！！！！！！！！！！！！！！！！！！！
# 我们之前介绍的深度可分离卷积是先做逐通道卷积，再做逐点卷积，而在Xception的论文描述中，这两步的顺序正好相反（见下图）。
# 不过没关系，论文中也指出，这里的顺序并不影响效果
# （理由：in particular because these operations are meant to be used in a stacked setting.）
#-----------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------#
# 这里其实是对应网络结构中的Conv
# 其中1*1卷积对应的padding为0，3*3卷积对应的padding为1，通过卷积操作的参数列表的条件判断得出
#--------------------------------------------------------------------------------#
def ConvBN(in_channels, out_channels, kernel_size,stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0 if kernel_size == 1 else (kernel_size-1)//2),
        nn.BatchNorm2d(out_channels),
    )

# 用于entryflow最上方的两个块
def ConvBNRelu(in_channels, out_channels, kernel_size,stride):
    return nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size, stride),
        nn.ReLU6(inplace=True),
    )

# 原文使用的是先depthwise，再pointwise
# 此处是先pointwise，再dwepthwise，但是对模型效果并无影响，对应网络结构中的SeperableConv
def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    )


# 对应网络结构中的SeparableConv + ReLU
def SeparableConvolutionRelu(in_channels, out_channels):
    return nn.Sequential(
        SeparableConvolution(in_channels, out_channels),
        nn.ReLU6(inplace=True),
    )

# 对应网络结构中的ReLU + SeparableConv
def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU6(inplace=True),
        SeparableConvolution(in_channels, out_channels)
    )

#------------------------------------------------------------------------------------------------------------------------------#
# 此处定义的bottleneck包含entryflow中的ReLU(由标志参数first_relu决定是否有) + SeparableConv + ReLU + SeparableConv + MaxPooling组成
# 此处仅对参数列表中的first_relu进行说明，其余两个参数是人就懂
# 引入标志参数first_relu的原因是，在entryflow的第一个bottleneck斌没有率先使用激活函数，与后续的bottleneck的结构有所不同
#------------------------------------------------------------------------------------------------------------------------------#
class EntryBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        # 中间产生的featuremap对应的通道数应该和out_channels一致
        mid_channels = out_channels

        # entryflow中shortcut对应1个1*1卷积的下采样路径
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels) if first_relu else SeparableConvolution(in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x

#------------------------------------------------------------------------------------------------------------------------------#
# 此处定义的bottleneck包含middleflow中的ReLU + SeparableConv + ReLU + SeparableConv + ReLU + SeparableConv组成
# 在middleflow中的bottleneck的残差边采用的是恒等映射，在前向传播函数中有所体现
#------------------------------------------------------------------------------------------------------------------------------#
class MiddleBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        return out + x

#-------------------------------------------------------------------------------------------------------------#
# 此处定义的bottleneck包含exitflow中的ReLU + SeparableConv + ReLU + SeparableConv + MaxPooling组成
#-------------------------------------------------------------------------------------------------------------#
class ExitBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels
        # exitflow中shortcut也对应1个1*1卷积的下采样路径
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        self.entryFlow = nn.Sequential(
            ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            EntryBottleneck(in_channels=64, out_channels=128, first_relu=False),
            EntryBottleneck(in_channels=128, out_channels=256, first_relu=True),
            EntryBottleneck(in_channels=256, out_channels=728, first_relu=True),
        )
        self.middleFlow = nn.Sequential(
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
        )
        self.exitFlow = nn.Sequential(
            ExitBottleneck(in_channels=728, out_channels=1024),
            SeparableConvolutionRelu(in_channels=1024, out_channels=1536),
            SeparableConvolutionRelu(in_channels=1536, out_channels=2048),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entryFlow(x)
        x = self.middleFlow(x)
        x = self.exitFlow(x)
        #-----------------------------------------------------------------------#
        # 以下是关于x = x.view(x.size(0), -1)的一些说明
        # 假设 x 的形状为 (batch_size, channels, height, width)
        # 那么这行代码将其形状改变为 (batch_size, channels * height * width) 的形式
        # 其中 channels * height * width 是一个一维向量，包含了所有的元素
        #-----------------------------------------------------------------------#
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


if __name__ == '__main__':
    model = Xception()
    print(model)

    input = torch.randn(1,3,299,299)
    output = model(input)
    print(output.shape)
