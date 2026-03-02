"""
多模态时空特征学习的领域自适应模型
包含注意力机制、图卷积网络和领域对齐组件
# -*- coding: utf-8 -*-
# @Time    : 2026/01/28
# @Author  : Yi Yang
# @File    : dnn_with_contrastive_learning_fixed.py
# @Result  : 加入对比学习提升类内紧凑性和类间分离性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channel, reduction=16):
        """
        初始化通道注意力模块

        参数:
            channel: 输入通道数
            reduction: 降维比例
        """
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.se = nn.Sequential(  # SE模块
            nn.Conv2d(channel, channel // reduction, 1, bias=False),  # 降维卷积
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)  # 升维卷积
        )
        self.sigmoid = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 [batch, channels, height, width]
        返回:
            通道注意力权重
        """
        max_result = self.maxpool(x)  # 最大池化特征
        avg_result = self.avgpool(x)  # 平均池化特征
        max_out = self.se(max_result)  # 处理最大特征
        avg_out = self.se(avg_result)  # 处理平均特征
        output = self.sigmoid(max_out + avg_out)  # 合并并激活
        return output


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块

        参数:
            kernel_size: 卷积核大小
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 [batch, channels, height, width]
        返回:
            空间注意力权重
        """
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度最大池化
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通道维度平均池化
        result = torch.cat([max_result, avg_result], 1)  # 拼接特征
        output = self.conv(result)  # 卷积融合
        output = self.sigmoid(output)  # 激活
        return output


class CBAMBlock(nn.Module):
    """卷积块注意力模块 (CBAM)"""

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        """
        初始化CBAM模块

        参数:
            channel: 通道数
            reduction: 通道注意力降维比例
            kernel_size: 空间注意力卷积核大小
        """
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量
        返回:
            (增强后的特征, 通道注意力, 空间注意力)
        """
        residual = x  # 残差连接
        ca = self.ca(x)  # 通道注意力
        out = x * ca  # 应用通道注意力
        sa = self.sa(out)  # 空间注意力
        out = out * sa  # 应用空间注意力
        return out + residual, ca, sa  # 残差连接


class Diffusion_GCN(nn.Module):
    """扩散图卷积网络"""

    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        """
        初始化扩散GCN

        参数:
            channels: 特征通道数
            diffusion_step: 扩散步数
            dropout: dropout率
        """
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        前向传播

        参数:
            x: 节点特征 [batch, channels, nodes, time]
            adj: 邻接矩阵 [batch, nodes, nodes] 或 [nodes, nodes]
        返回:
            扩散后的特征
        """
        out = []
        for _ in range(self.diffusion_step):
            if adj.dim() == 3:  # 批量处理
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
            elif adj.dim() == 2:  # 共享邻接矩阵
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
            out.append(x)

        x = torch.cat(out, dim=1)  # 拼接所有扩散步的结果
        x = self.conv(x)  # 特征融合
        return self.dropout(x)


class Graph_Generator(nn.Module):
    """动态图生成器"""

    def __init__(self, channels=5, num_nodes=62, dropout=0.1):
        """
        初始化图生成器

        参数:
            channels: 特征通道数
            num_nodes: 节点数量
            dropout: dropout率
        """
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))  # 记忆参数
        nn.init.xavier_uniform_(self.memory)  # 参数初始化
        self.fc = nn.Linear(2, 1)  # 融合层

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch, channels, nodes, time]
        返回:
            动态邻接矩阵
        """
        # 基于记忆的相似性
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )

        # 基于节点特征的相似性
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )

        # 融合两种相似性
        adj_f = torch.cat([adj_dyn_1.unsqueeze(-1), adj_dyn_2.unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        # 保留top-k连接
        k = int(adj_f.shape[1] * 0.8)  # 保留80%的连接
        topk_values, topk_indices = torch.topk(adj_f, k=k, dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask

        return adj_f


class DGCN(nn.Module):
    """动态图卷积网络"""

    def __init__(self, channels=5, num_nodes=62, diffusion_step=1, dropout=0.1):
        """
        初始化DGCN

        参数:
            channels: 通道数
            num_nodes: 节点数
            diffusion_step: 扩散步数
            dropout: dropout率
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.generator = Graph_Generator(channels, num_nodes, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征
        返回:
            (增强特征, 动态邻接矩阵)
        """
        skip = x
        x = self.conv(x)  # 特征变换
        adj_dyn = self.generator(x)  # 生成动态图
        x = self.gcn(x, adj_dyn)  # 图卷积
        return x + skip, adj_dyn  # 残差连接


class GATENet(nn.Module):
    """图注意力编码网络"""

    def __init__(self, inc, chan, hidden, reduction_ratio=128):
        """
        初始化GATENet

        参数:
            inc: 输入维度
            chan: 通道数
            hidden: 隐藏层维度
            reduction_ratio: 降维比例
        """
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inc, inc // reduction_ratio, bias=False),
            nn.ELU(inplace=False),
            nn.Linear(inc // reduction_ratio, chan * hidden, bias=False),
            nn.Tanh(),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """前向传播"""
        return self.fc(x)


class MHGCN(nn.Module):
    """多尺度层次图卷积网络"""

    def __init__(self, layers, dim, chan_num, band_num, hidden_1, hidden_2):
        """
        初始化MHGCN

        参数:
            layers: 层数
            dim: 特征维度
            chan_num: 通道数
            band_num: 频带数
            hidden_1: 隐藏层1维度
            hidden_2: 隐藏层2维度
        """
        super(MHGCN, self).__init__()
        self.chan_num = chan_num
        self.band_num = band_num
        self.hidden = 5
        self.A = torch.rand((1, self.chan_num * self.chan_num),
                            dtype=torch.float32, requires_grad=False)

        self.GATENet = GATENet(self.chan_num * self.chan_num, self.chan_num,
                               self.hidden, reduction_ratio=128)

        # 构建多层DGCN
        self.HGCN_layers = nn.ModuleList()
        for _ in range(layers):
            self.HGCN_layers.append(DGCN(channels=5, num_nodes=62,
                                         diffusion_step=1, dropout=0.1))

        self.initialize()

    def initialize(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                    if isinstance(j, nn.Linear):
                        nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征
        返回:
            (多层特征融合, 邻接矩阵列表)
        """
        output = []
        A = []
        output.append(x)

        # 逐层处理
        for layer in self.HGCN_layers:
            out = layer(x)
            output.append(out[0])
            A.append(out[1])
            x = output[-1]

        out = torch.cat(output, dim=1)  # 拼接所有层输出
        return out, A


class Encoder(nn.Module):
    """编码器模块"""

    def __init__(self, in_planes=[5, 62], layers=2, hidden_1=256,
                 hidden_2=64, class_nums=3):
        """
        初始化编码器

        参数:
            in_planes: 输入维度 [频带数, 通道数]
            layers: 图卷积层数
            hidden_1: 隐藏层1维度
            hidden_2: 隐藏层2维度
            class_nums: 类别数
        """
        super(Encoder, self).__init__()
        self.chan_num = in_planes[1]
        self.band_num = in_planes[0]

        # 多尺度图卷积网络
        self.GGCN = MHGCN(layers=layers, dim=1, chan_num=self.chan_num,
                          band_num=self.band_num, hidden_1=hidden_1, hidden_2=hidden_2)

        # 注意力模块
        self.CBAM = CBAMBlock(channel=(layers + 1) * self.band_num,
                              reduction=4, kernel_size=3)

        # 全连接层
        self.fc1 = nn.Linear(self.chan_num * (layers + 1) * self.band_num, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量
        返回:
            (编码特征, [邻接矩阵, 通道注意力, 空间注意力])
        """
        # 调整输入形状
        x = x.reshape(x.size(0), 5, 62)
        x = x.unsqueeze(3)

        # 图卷积特征提取
        g_feat, g_adj = self.GGCN(x)

        # 注意力机制
        g_feat, ca, sa = self.CBAM(g_feat)

        # 全连接层
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        return out, [g_adj, ca, sa]


class ClassClassifier(nn.Module):
    """分类器"""

    def __init__(self, hidden_2, num_cls):
        """
        初始化分类器

        参数:
            hidden_2: 输入特征维度
            num_cls: 类别数
        """
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_2, num_cls)

    def forward(self, x):
        """前向传播"""
        return self.classifier(x)


class DomainAdaptationModel(nn.Module):
    """领域自适应模型"""

    def __init__(self, in_planes=[5, 62], layers=2, hidden_1=256,
                 hidden_2=64, num_of_class=3, device='cuda:0',
                 source_num=3944, target_num=851):
        """
        初始化领域自适应模型

        参数:
            in_planes: 输入维度
            layers: 网络层数
            hidden_1: 隐藏层1维度
            hidden_2: 隐藏层2维度
            num_of_class: 类别数
            device: 计算设备
            source_num: 源域样本数
            target_num: 目标域样本数
        """
        super(DomainAdaptationModel, self).__init__()

        # 组件初始化
        self.encoder = Encoder(in_planes=in_planes, layers=layers,
                               hidden_1=hidden_1, hidden_2=hidden_2,
                               class_nums=num_of_class)
        self.cls_classifier = ClassClassifier(hidden_2=hidden_2,
                                              num_cls=num_of_class)

        # 内存存储
        self.source_f_bank = torch.zeros(source_num, hidden_2)
        self.target_f_bank = torch.zeros(target_num, hidden_2)
        self.source_score_bank = torch.zeros(source_num, num_of_class).to(device)
        self.target_score_bank = torch.zeros(target_num, num_of_class).to(device)
        self.source_label_bank = torch.full((source_num,), -1, dtype=torch.long)

        # 超参数
        self.num_of_class = num_of_class
        self.ema_factor = 0.8
        self.tem = 1
        self.device = device

    def forward(self, source, target, source_label,
                source_index, target_index, current_epoch, max_epochs):
        """
        前向传播

        参数:
            source: 源域数据
            target: 目标域数据
            source_label: 源域标签
            source_index: 源域样本索引
            target_index: 目标域样本索引
            current_epoch: 当前训练轮次
            max_epochs: 最大训练轮次
        返回:
            包含各种特征的元组
        """
        # 编码特征
        source_f, [self.src_adj, self.src_sa, self.src_ca] = self.encoder(source)
        target_f, [self.tar_adj, self.tar_sa, self.tar_ca] = self.encoder(target)

        # 分类预测
        source_predict = self.cls_classifier(source_f)
        target_predict = self.cls_classifier(target_f)

        # 获取概率分布
        source_label_feature = F.softmax(source_predict, dim=1)
        target_label_feature = F.softmax(target_predict, dim=1)

        # 相似性计算
        src_sim, src_prototype = self._get_source_similar(source_f,
                                                          source_label_feature,
                                                          source_index)

        tgt_sim, tgt_prototype, tat_cluster_label = self._get_target_similar(
            target_f, target_label_feature, target_index,
            src_prototype, current_epoch, max_epochs
        )

        # 跨域相似性
        s2t_pro = self._get_st_similar(source_f, tgt_prototype)
        t2s_pro = self._get_st_similar(target_f, src_prototype)
        s2s_pro = self._get_st_similar(source_f, src_prototype)
        t2t_pro = self._get_st_similar(target_f, tgt_prototype)

        return (source_predict, source_f, target_predict, target_f,
                [self.src_adj, self.src_sa, self.src_ca],
                [self.tar_adj, self.tar_sa, self.tar_ca],
                src_sim, tgt_sim, tat_cluster_label,
                s2t_pro, t2s_pro, s2s_pro, t2t_pro)

    def _get_source_similar(self, feature_source_f, source_label_feature, source_index):
        """
        计算源域相似性

        参数:
            feature_source_f: 源域特征
            source_label_feature: 源域预测分布
            source_index: 源域样本索引
        返回:
            (相似性矩阵, 类别原型)
        """
        self.eval()
        # 特征归一化
        output_f = F.normalize(feature_source_f, p=2, dim=1)

        # 更新特征和得分库
        self.source_f_bank[source_index] = output_f.detach().clone().cpu()
        self.source_score_bank[source_index] = source_label_feature.detach().clone()

        # 计算类别原型
        prototype_class = []
        for class_id in range(self.num_of_class):
            # 获取当前类别的预测标签
            pred_labels = torch.argmax(self.source_score_bank, dim=1).cpu()
            source_feature = self.source_f_bank[pred_labels == class_id]

            if source_feature.size(0) > 0:
                prototype = source_feature.mean(dim=0)
            else:
                # 处理空类别的情况
                prototype = torch.zeros(output_f.size(1), device=source_feature.device)

            prototype_class.append(prototype)

        prototypes = torch.stack(prototype_class)  # [num_classes, feature_dim]

        # 计算相似性
        src_sim = torch.mm(
            output_f.to(self.device),
            F.normalize(prototypes.to(self.device), p=2, dim=1).T
        ) / self.tem

        return src_sim, prototypes

    def _get_target_similar(self, feature_target_f, target_label_feature,
                            target_index, src_prototype, current_epoch, max_epochs):
        """
        计算目标域相似性

        参数:
            feature_target_f: 目标域特征
            target_label_feature: 目标域预测分布
            target_index: 目标域样本索引
            src_prototype: 源域原型
            current_epoch: 当前轮次
            max_epochs: 总轮次
        返回:
            (相似性矩阵, 目标域原型, 聚类标签)
        """
        self.eval()
        # 特征归一化
        f = F.normalize(feature_target_f, p=2, dim=1)

        # 更新目标域特征库
        self.target_f_bank[target_index] = f.detach().clone().cpu()
        self.target_score_bank[target_index] = target_label_feature.detach().clone()

        # 选择高置信度样本
        output = self.target_f_bank.to(self.device)
        scores = self.target_score_bank
        aggregated_scores = scores.max(dim=1)[0]
        num_samples = len(aggregated_scores)

        # 选择top-k高置信度样本
        k = int(num_samples * 0.3)  # 选择30%的样本
        _, top_indices = torch.topk(aggregated_scores, k)
        output_f = output[top_indices]

        # K-means聚类
        kmeans = KMeans(n_clusters=self.num_of_class, random_state=0)
        kmeans.fit(output_f.cpu().detach().numpy())
        prototype = torch.tensor(kmeans.cluster_centers_, device=self.device)

        # 计算相似性
        tgt_sim = torch.mm(
            F.normalize(f, p=2, dim=1),
            F.normalize(prototype, p=2, dim=1).T
        ) / self.tem

        # 获取伪标签
        target_predict = F.softmax(tgt_sim, dim=1)
        tar_label = torch.argmax(target_predict, dim=1)

        return tgt_sim, prototype, tar_label

    def _get_st_similar(self, feature, prototypes):
        """
        计算源域-目标域相似性

        参数:
            feature: 特征
            prototypes: 原型向量
        返回:
            相似性概率分布
        """
        if prototypes.numel() == 0:
            return torch.zeros((feature.size(0), 3), device=feature.device)

        feature = F.normalize(feature, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        st_sim = torch.mm(
            feature.to(self.device),
            prototypes.to(self.device).T
        ) / self.tem

        return F.softmax(st_sim, dim=1)

    def get_init_banks(self, source, source_index):
        """
        初始化源域特征库

        参数:
            source: 源域数据
            source_index: 源域样本索引
        """
        self.eval()
        with torch.no_grad():
            source_f, _ = self.encoder(source)
            source_predict = self.cls_classifier(source_f)
            source_label_feature = F.softmax(source_predict, dim=1)

            self.source_f_bank[source_index] = F.normalize(source_f).detach().clone().cpu()
            self.source_score_bank[source_index] = source_label_feature.detach().clone()

    def get_init_banks_tgt(self, tgt, tgt_index):
        """
        初始化目标域特征库

        参数:
            tgt: 目标域数据
            tgt_index: 目标域样本索引
        """
        self.eval()
        with torch.no_grad():
            tgt_f, _ = self.encoder(tgt)
            tgt_predict = self.cls_classifier(tgt_f)
            tgt_label_feature = F.softmax(tgt_predict, dim=1)

            self.target_f_bank[tgt_index] = F.normalize(tgt_f).detach().clone().cpu()
            self.target_score_bank[tgt_index] = tgt_label_feature.detach().clone()

    def target_predict(self, feature_target):
        """
        目标域预测

        参数:
            feature_target: 目标域特征
        返回:
            目标域预测分布
        """
        self.eval()
        with torch.no_grad():
            target_f, _ = self.encoder(feature_target)
            target_predict = self.cls_classifier(target_f)
            return F.softmax(target_predict, dim=1)

    def domain_discrepancy(self, out1, out2, loss_type='L2'):
        """
        计算域间差异损失

        参数:
            out1: 域1特征
            out2: 域2特征
            loss_type: 损失类型 ('L1', 'Huber', 'L2')
        返回:
            域差异损失
        """

        def huber_loss(e, d=1):
            """Huber损失函数"""
            t = torch.abs(e)
            ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
            return torch.mean(ret)

        diff = out1 - out2
        if loss_type == 'L1':
            loss = torch.mean(torch.abs(diff))
        elif loss_type == 'Huber':
            loss = huber_loss(diff)
        else:  # L2损失
            loss = torch.mean(diff * diff)

        return loss

    def get_weight(self, score_near):
        """
        计算基于熵的权重

        参数:
            score_near: 邻近样本得分
        返回:
            归一化的权重
        """
        epsilon = 1e-5
        entropy = -(1 / score_near.size(1)) * torch.sum(score_near * torch.log(score_near + epsilon), dim=2)
        g = 1 - entropy
        score_near_weight = g / torch.tile(torch.sum(g, dim=1).view(-1, 1), (1, score_near.size(1)))
        return score_near_weight

    def entropy(self, input_):
        """
        计算熵

        参数:
            input_: 输入概率分布
        返回:
            熵值
        """
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        return torch.sum(entropy, dim=1)


# 模型使用示例
if __name__ == "__main__":
    # 模型参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    # 创建模型
    model = DomainAdaptationModel(
        in_planes=[5, 62],
        layers=2,
        hidden_1=256,
        hidden_2=64,
        num_of_class=3,
        device=device
    ).to(device)

    # 创建模拟数据
    source_data = torch.randn(batch_size, 5 * 62).to(device)
    target_data = torch.randn(batch_size, 5 * 62).to(device)
    source_labels = torch.randint(0, 3, (batch_size,)).to(device)
    source_indices = torch.randint(0, 3944, (batch_size,))
    target_indices = torch.randint(0, 851, (batch_size,))

    # 前向传播
    outputs = model(
        source_data, target_data, source_labels,
        source_indices, target_indices, 1, 100
    )

    print("模型输出包含以下内容:")
    print(f"1. 源域预测形状: {outputs[0].shape}")
    print(f"2. 源域特征形状: {outputs[1].shape}")
    print(f"3. 目标域预测形状: {outputs[2].shape}")
    print(f"4. 目标域特征形状: {outputs[3].shape}")