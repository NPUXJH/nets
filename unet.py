import torch
import torch.nn as nn

'''
使用unet的网络结构，但是需要注意unet原文中，对特征层做卷积时，没有用padding做填充，但本文中需要填充
'''
class DoubleConvBR(nn.Module):
    # 两个卷积 + 批标准化 + ReLU
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleConvbr_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.doubleConvbr_block(x)

class DownSample(nn.Module):
    #下采样块：一次最大池化 + 一个DoubleConvBR
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downSample_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvBR(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downSample_block(x)

class UpSample(nn.Module):
    # 上采样块：先通过一个1*1卷积调整通道数（减半），再进行上采样扩大尺寸，与对应的下采样的特征层做堆叠操作
    # 再加一个DoubleConvBR
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upSample_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.upSample_conv = nn.Sequential(
            DoubleConvBR(in_channels, out_channels)
        )

    def forward(self, feature_map1, feature_map2):
        feature_map1 = self.upSample_up(feature_map1)
        concat_map = torch.cat(feature_map1, feature_map2)
        return self.upSample_conv(concat_map)


class OutputConv(nn.Module):
    # 输出结果，语义分割中通过卷积操作将输出通道数调整到对应的类别
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.outputConv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=1)
        )

    def forward(self, x):
        return self.outputConv_block(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):

        super(Unet,self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.start = DoubleConvBR(n_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.output = OutputConv(64, n_classes)

    def forward(self, result):
        x1 = self.start(result)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output(x)

if __name__ == '__main__':
    model = Unet(3,100)
    print(model)