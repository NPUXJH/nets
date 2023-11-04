import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __int__(self, embedding_net):
        super(SiameseNet, self).__init__()
        # 嵌入网络，用于将输入数据转为嵌入表示
        self.embedding_net = embedding_net
        # 1表示归一化的特征通道数
        self.match_batchnorm = nn.BatchNorm2d(1)

    # 定义前向传播函数
    def forward(self, x1, x2):
        # 对参考输入进行嵌入表示
        embedding_reference = self.embedding_net(x1)
        # 对搜索输入及逆行嵌入式表示
        embedding_search = self.embedding_net(x2)
        # 计算匹配图
        match_map = self.match_corr(embedding_reference, embedding_search)
        return  match_map

    def get_response(self, x, kernel):
        # 对输入进行嵌入式表示
        ebx = self.embedding_net(x)
        # 计算匹配图
        return self.match_corr(kernel, ebx)

    # 定义匹配图计算的函数
    def match_corr(self, reference, search):
        # 获得search图像的尺寸信息，b是批次大小，c是通道数，h是高度，w是宽度
        b, c, h, w = search.shape
        # 相关操作，计算匹配图
        # search.view(1, b*c, h, w)是search图片调整形状后的张量
        # 通过groups=b将输入的search图像分为b个组，对应每个search图像，分别进行卷积
        match_map = F.conv2d(search.view(1, b*c, h, w), reference, groups=b)
        # 对match_map的维度重新进行排列操作
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        return match_map
