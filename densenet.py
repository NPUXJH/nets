import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


#---------------------------------------------------------------------------------------------------------#
# num_input_features: 这是_DenseLayer层的输入特征数量
# growth_rate: growth_rate是每个_DenseLayer层输出的特征图数量
# bn_size: bn_size是用于控制每个_DenseLayer层内部瓶颈层的通道数，是1*1卷积将为后的特征层数，在原论文中，这个值通常取4
# drop_rate: drop_rate是一个浮点数，表示在_DenseLayer的输出上应用的丢弃率
# 注意此处卷积部分采用了BN-ReLU-Conv的卷积结构
#---------------------------------------------------------------------------------------------------------#
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    #--------------------------------------------------------------------------#
    # 这里表示是否进行正则化，当drop_rate>0且模型不处于训练状态时，则使用dropout正则化
    # 通常在卷积层中，不同时使用BN和dropout正则化，因为可能造成冲突
    #--------------------------------------------------------------------------#
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # 将原始输入x和经过处理的new_features在通道维度上（维度 1）进行拼接
        return torch.cat([x, new_features], 1)


#------------------------------------------------------------------------------------------#
# denseblock是由多个denselayer层构成，代码通过循环添加多个_DenseLayer实例，形成DenseBlock
# num_layers: 这是整数参数，表示在_DenseBlock中要堆叠的_DenseLayer层的数量
# num_input_features: 这是整数参数，表示输入到_DenseBlock的特征图的通道数
# 其余参数与上述denselayer的参数含义相同，此处不再赘述
#------------------------------------------------------------------------------------------#
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        # i从0开始
        for i in range(num_layers):
            # 这里就表明了在denseblock结构中，每层的输入特征层数是由进入denseblock的层数和之前所有的denselayer的层数之和
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            # 这是添加的子模块的名称。在这里，子模块的名称采用了一个格式化字符串，其中 %d 会被替换为 (i + 1)，以创建不同层的不同名称，如 'denselayer1'
            self.add_module('denselayer%d' % (i + 1), layer)


#-------------------------------------------------------------------------------------------#
# 下面时densenet结构中transition layer的定义，这是一个下采样路径
# 通过一个1*1卷积和一个步长为2的2*2的全局平均池化来实现下采样
# 依然采用了BN-ReLU-Conv的卷积结构
#-------------------------------------------------------------------------------------------#
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        #---------------------------------------------------------------------------------------------------------------#
        # 这里是所有densenet结构都要进行的卷积处理，它是网络刚开始的部分，实现了两次下采样
        # 先使用一个步长为2的7*7卷积对图像进行处理，注意，此时要求将图像尺寸严格缩放到原来的一半，所以此处对图像边缘加了paddinng=3的填充
        # 然后使用了一个步长为2的3*3的最大池化进行下采样
        # --------------------------------------------------------------------------------------------------------------#
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        #------------------------------------------------------------------------------------------------------------#
        # 此处是denselayer的核心结构，用于将denseblock和transition layer结构堆叠起来
        #------------------------------------------------------------------------------------------------------------#

        # num_features表示传入denseblock的特征层数
        num_features = num_init_features
        # 遍历block_config的索引和元素，i+1表示denseblock的序号（从1开始），其对应的元素表示每个denseblock中denselayer的数量
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # 在不是最后一个denseblock后都要插入一个transition layer
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                # transition layer进行下采样的同时也会压缩通道数
                # transition layer压缩通道数的超参数是compression rate，为0至1区间内的小数，常取0.5
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        #----------------------------------------------------------------------------#
        # Official init from torch repo.
        # 这个部分是用于对模型的参数进行初始化的代码
        # 对于卷积层(nn.Conv2d)，采用了He初始化(nn.init.kaiming_normal)，这是一种常用的权重初始化策略，有助于训练深度神经网络
        # 对于批归一化层 (nn.BatchNorm2d)，将权重初始化为1，偏置初始化为0，这是一种常用的初始化策略，有助于网络的稳定训练
        # 对于线性层(nn.Linear)，将偏置初始化为0
        #----------------------------------------------------------------------------#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    #----------------------------------------------------------------------------------------------------------------------------#
    # 以下定义了densenet的前向传播方法
    # 将网络提取到的特征图应用ReLU激活，inplace=True 表示原地操作，即在 features 上直接修改，这有助于减少内存开销
    # 然后用一个7*7的全局平均池化来代替传统的卷积神经网络中的全连接层，有助于减少参数量
    # 然后，使用 .view 方法将特征图的形状从 (batch_size, num_features, 1, 1) 转换为 (batch_size, num_features)，以便输入到分类器（全连接层）
    # 将全局平均池化后的特征out输入到分类器（全连接层）self.classifier中，以生成模型的最终输出
    #----------------------------------------------------------------------------------------------------------------------------#
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # features.size(0) 表示批量大小（batch size），而 -1 表示自动计算该维度的大小
        # 这样，平均池化后的 1x1 特征图被展平成一个一维向量，其中每个元素对应一个特征
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# densenet121:每个densseblock的denselayer数依次为6，12，24，16
def densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model

# densenet169:每个densseblock的denselayer数依次为6，12，32，32
def densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model

# densenet201:每个densseblock的denselayer数依次为6，12，48，32
def densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model

# densenet161:每个densseblock的denselayer数依次为6，12，36，24
def densenet161(**kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model


if __name__ == '__main__':
    # ‘DenseNet‘, ‘densenet121‘, ‘densenet169‘, ‘densenet201‘, ‘densenet161‘
    # Example
    net = DenseNet()
    print(net)
