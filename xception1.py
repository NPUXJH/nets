import torch
import math
import torch.nn as nn
#------------------------------------------------------------------------------------------------#
# xcetion网络架构由深度学习框架keras的作者Francois Chollet提出
# ineption系列自2014年GoogleNet（inception v1）问世以来，inception系列持续发展
# 随后出现了inception-BN,inception v2,inception-v3,inception-v4,inception-resnet,xception等网络结构
# 后续的inception v3和xception等网络被广泛用于迁移学习的基网络
# 参考链接：https://www.modb.pro/db/178156
#------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------#
# Notice!!!!!!!!!!!!!!!!!!!!!!!!!!
# 我们之前介绍的深度可分离卷积是先做逐通道卷积，再做逐点卷积，而在Xception的论文描述中，这两步的顺序正好相反（见下图）。
# 不过没关系，论文中也指出，这里的顺序并不影响效果
# （理由：in particular because these operations are meant to be used in a stacked setting.）
#-----------------------------------------------------------------------------------------------------#

# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        # 逐点卷积：普通1x1卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        #:parm reps:块重复次数
        super(Block, self).__init__()

        # Middle flow无需做这一步，而其余块需要，以做跳连
        # 1）Middle flow输入输出特征图个数始终一致，且stride恒为1
        # 2）其余块需要stride=2，这样可以将特征图尺寸减半，获得与最大池化减半特征图尺寸同样的效果
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # Middle flow 的stride恒为1，因此无需做池化，而其余块需要
        # 其余块的stride=2，因此这里的最大池化可以将特征图尺寸减半
        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes  # 总分类数

        ################################## 定义 Entry flow ###############################################################
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        ################################### 定义 Middle flow ############################################################
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        #################################### 定义 Exit flow ###############################################################
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        ###################################################################################################################

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # 定义Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # 定义 Exit flow
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = Xception()
    print(model)