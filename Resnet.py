import torch
import torch.nn as nn

#-------------------------------------------------------------------------------------------------#
# 本代码主要搭建Kaiming He在2015发表的Deep Residual Learrning for Image Recognition中提出的深度残差网络
# 本代码中构建了原论文中提到的resnet-18,resnet-34,resnet-50,resnet-101,resnet-152四种网络结构
# 其中，由于网络结构的差异，resnet-18和resnet-34使用一种卷积块，其余使用另一种卷积块
# 较浅层的卷积块名为ShallowConvBlock，较深层的卷积块名为DeepConvBlock
# 参考博客：https://blog.csdn.net/frighting_ing/article/details/121324000
#-------------------------------------------------------------------------------------------------#

# 以下定义ResNet-18/34 残差结构 ShallowConvBlock
# ResNet-18/34 残差结构 BasicBlock
class BasicBlock(nn.Module):
    # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1
    expansion = 1
    #------------------------------------------------------------------------------------#
    # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
    # stride默认为1，不压缩尺寸
    # downsample对应虚线残差结构捷径中的1×1卷积
    #------------------------------------------------------------------------------------#
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使用bn层时不使用bias
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 实/虚线残差结构主分支中第二层stride都为1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 默认是None
        self.downsample = downsample

    # 定义正向传播过程
    def forward(self, x):
        # 捷径分支的输出值
        identity = x
        # 对应虚线残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 这里不经过relu激活函数
        out += identity
        out = self.relu(out)
        return out


# ResNet-50/101/152 残差结构 Bottleneck
class Bottleneck(nn.Module):
    #-----------------------------------------------------------------------------------------------------#
    # 注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    # 但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    # 这么做的好处是能够在top1上提升大概0.5%的准确率。
    # 只有conv3_x,conv4_x,conv5_x的第一个块的3*3卷积有stride=2的下采样路径，downsample不只调整虚线的shortcut的尺寸，也可以用来调整通道数
    # 可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    #-----------------------------------------------------------------------------------------------------#

    # 第三层的卷积核个数是第一层、第二层的四倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # out_channels是第一、二层的卷积核个数，squeeze channels  高和宽不变
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -------------------------------------------
        # 实线stride为1，虚线stride为2
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -------------------------------------------
        # 卷积核个数为4倍 unsqueeze channels
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 正向传播过程
    def forward(self, x):
        identity = x
        # 对应虚线残差结构
        if self.downsample is not None:
            identity = self.downsample(x)
        # 第一个1*1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3*3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 第二个1*1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


# ResNet整个网络的框架部分
class ResNet(nn.Module):
    #----------------------------------------------------------------------#
    # 参数说明：
    # block:残差结构，Basicblock or Bottleneck
    # block_num:列表参数，所使用残差结构的数目，如对ResNet-34来说即是[3,4,6,3]
    # num_classes:训练集的分类个数
    # include_top=True:# 为了能在ResNet网络基础上搭建更加复杂的网络，默认为True
    # ---------------------------------------------------------------------#
    def __init__(self, block, blocks_num, num_classes=1000,  include_top=True):
        super(ResNet, self).__init__()

        # 传入类变量
        self.include_top = include_top
        # 通过max pooling之后所得到的特征矩阵的深度
        self.in_channel = 64
        # 输入特征矩阵的深度为3（RGB图像），高和宽缩减为原来的一半
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 高和宽缩减为原来的一半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 对应conv2_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        # 对应conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # 对应conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # 对应conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 默认为True
        if self.include_top:
            # 无论输入特征矩阵的高和宽是多少，通过自适应平均池化下采样层，所得到的高和宽都是1
            # output size = (1, 1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # num_classes为分类类别数
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # num_classes为分类类别数

        # 卷积层的初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #-------------------------------------------------------------#
    # block即BasicBlock/Bottleneck
    # channel即残差结构中第一层卷积层所使用的卷积核的个数
    # block_num即该层一共包含了多少层残差结构
    # stride默认为1
    #-------------------------------------------------------------#

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # ------------------------------------------------------------------#
        # 左：输出的高和宽相较于输入会缩小（下采样）；右：输入channel数与输出channel数不相等
        # 两者都会使x和identity无法相加
        # ResNet-18/34会直接跳过该if语句（对于layer1来说）
        # 对于ResNet-50/101/152：
        # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
        # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
        # 下采样
        # 将特征矩阵的深度翻4倍，高和宽不变（对于layer1来说）
        # ------------------------------------------------------------------#
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        #--------------------------------------------------------------#
        # self.in_channel:输入特征矩阵深度，64
        # channel:残差结构所对应主分支上的第一个卷积层的卷积核个数
        #--------------------------------------------------------------#
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        # 更新第一个block后的同组block的输入通道数
        self.in_channel = channel * block.expansion

        #---------------------------------------------------------------#
        # 从第二层开始都是实线残差结构
        # 对于浅层一直是64，对于深层已经是64*4=256了
        # 残差结构主分支上的第一层卷积的卷积核个数
        #---------------------------------------------------------------#
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        # 通过非关键字参数的形式传入nn.Sequential
        # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回
        return nn.Sequential(*layers)

    # 正向传播过程
    def forward(self, x):
        # 7×7卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 3×3 max pool
        x = self.maxpool(x)

        # conv2_x所对应的一系列残差结构
        x = self.layer1(x)
        # conv3_x所对应的一系列残差结构
        x = self.layer2(x)
        # conv4_x所对应的一系列残差结构
        x = self.layer3(x)
        # conv5_x所对应的一系列残差结构
        x = self.layer4(x)

        if self.include_top:
            # 平均池化下采样
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

if __name__ == '__main__':
    # ‘DenseNet‘, ‘densenet121‘, ‘densenet169‘, ‘densenet201‘, ‘densenet161‘
    # Example
    net = resnet101()
    print(net)