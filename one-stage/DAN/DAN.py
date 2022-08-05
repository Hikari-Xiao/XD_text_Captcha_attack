# coding:utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
import resnet as resnet

'''
Feature_Extractor。DAN的第一部分：特征编码器，用于提取视觉特征。
'''


class Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape):
        """
        :param strides: 特征提取阶段各个块的步长
        :param compress_layer: 是否有压缩层
        :param input_shape: 输入图的shape
        """
        super(Feature_Extractor, self).__init__()
        self.model = resnet.resnet45(strides, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        """返回有高宽改变的几个块和最后一块的输出特征图组成的list，对于scene，即[layer1， layer3， layer5]"""
        features = self.model(input)
        return features

    def Iwantshapes(self):
        """
        给一个和输入同shape的随机输入，获取上述几个块的输出特征图的尺寸（C, H, W），不包含batchsize。
        示例：若输入通道、高、宽分别是(1, 32, 128)，
             则输出[(32, 16, 64),(128, 8, 32),(512, 8, 32)]
        """
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


'''
Convolutional Alignment Module。第二部分：卷积对齐模块（CAM），基于上一步中得到的视觉特征实现对齐操作。
'''


# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.
class CAM(nn.Module):
    def __init__(self, scales, maxT, depth, num_channels):
        """
        :param scales: 特征提取输出的几个块的特征图对应的尺寸（C, H, W）
        :param maxT: 注意图的数量（也是反卷积最后一层输出的特征通图道数，需要大于数据集文本最长长度即可）
        :param depth: CAM中的卷积和反卷积总共的层数（各depth/2层）。
        :param num_channels: CAM中除了最后一层外的所有层的通道数（均是num_channels）
        """
        super(CAM, self).__init__()
        # cascade multiscale features。将特征编码输出的多个尺度的特征图（不包含最后一块特征图）依次下采样卷积（conv2+BN+ReLu）
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]  # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(scales[i - 1][2] / scales[i][2])
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
                                               (ksize_h, ksize_w),
                                               (r_h, r_w),
                                               (int((ksize_h - 1) / 2), int((ksize_w - 1) / 2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)

        # convolutional alignment。卷积对齐模块，除最后一层的输出通道为maxT，其他均为统一的num_channel
        # convs。卷积对齐模块的卷积部分（depth/2层 conv2+BN+ReLU）
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]  # 特征编码（提取）最后一块输出特征图的尺寸（C, H, W）
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
            strides.append(stride)  # (1, 2), (2, 2), (2, 2), (2, 2)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                         tuple(conv_ksizes[0]),
                                         tuple(strides[0]),
                                         (int((conv_ksizes[0][0] - 1) / 2), int((conv_ksizes[0][1] - 1) / 2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                 tuple(conv_ksizes[i]),
                                                 tuple(strides[i]),
                                                 (int((conv_ksizes[i][0] - 1) / 2), int((conv_ksizes[i][1] - 1) / 2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)

        # deconvs。卷积对齐模块的反卷积部分（depth/2 - 1层 ‘转置卷积+BN+ReLU’ 和 1层 ‘转置卷积+sigmoid’）
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                            tuple(deconv_ksizes[int(depth / 2) - i]),
                                                            tuple(strides[int(depth / 2) - i]),
                                                            (int(deconv_ksizes[int(depth / 2) - i][0] / 4.),
                                                             int(deconv_ksizes[int(depth / 2) - i][1] / 4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                        tuple(deconv_ksizes[0]),
                                                        tuple(strides[0]),
                                                        (int(deconv_ksizes[0][0] / 4.), int(deconv_ksizes[0][1] / 4.))),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        # 依次将特征编码阶段每个块输出的特征图执行下采样卷积，然后与下一块的特征图输出相加
        x = input[0]    # shape(nB, 32, H/2, W/2)
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i + 1]    # shape(nB, 512, H/4, W/4)

        # 对上一步汇总所得的输出图依次执行len(self.convs)层卷积操作，并将每一次卷积后的输出存入conv_feats
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)  # shape(nB, 64, H/4, W/8)——(64, H/8, W/16)——(64, H/16, W/32)——(64, H/32, W/64)  (1, 2)
            conv_feats.append(x)

        # 对上一步输出执行几层反卷积操作，并将每一层输出与（正）卷积对应的同shape的输出相叠加（除过最后一层），输入下一层
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[len(conv_feats) - 2 - i]

        x = self.deconvs[-1](x)  # shape(nB, maxT, H/4, W/4)
        return x  # 返回注意力图，shape(N, maxT, H/4, W/4)，示例：若输入高宽32,128，则(nB, 25, 8, 32)


class CAM_transposed(nn.Module):
    # In this version, the input channel is reduced to 1-D with sigmoid activation.
    # We found that this leads to faster convergence for 1-D recognition.
    def __init__(self, scales, maxT, depth, num_channels):
        """
             :param scales: 特征提取的5个layer输出的特征图对应的尺寸（C, H, W）
             :param maxT: 注意图的数量（也是反卷积最后一层输出的特征通图道数）
             :param depth: CAM中的卷积和反卷积总共的层数（各depth/2层）。
             :param num_channels: CAM中除了最后一层外的所有层的通道数（均是128，为了涵盖最长的文本长度）
             """
        super(CAM_transposed, self).__init__()
        # cascade multiscale features。将特征编码输出的多个尺度的特征图（不包含最后一块特征图）依次下采样卷积（conv2+BN+ReLu）
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]
            r_h, r_w = scales[i - 1][1] / scales[i][1], scales[i - 1][2] / scales[i][2]
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
                                               (ksize_h, ksize_w),
                                               (r_h, r_w),
                                               ((ksize_h - 1) / 2, (ksize_w - 1) / 2)),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        fpn.append(nn.Sequential(nn.Conv2d(scales[i][0], 1,
                                           (1, ksize_w),
                                           (1, r_w),
                                           (0, (ksize_w - 1) / 2)),
                                 nn.Sigmoid()))   #
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment。卷积对齐模块，除最后一层的输出通道为maxT，其他均为统一的num_channel
        # deconvs。反卷积
        in_shape = scales[-1]
        deconvs = []
        ksize_h = 1 if in_shape[1] == 1 else 4
        for i in range(1, depth / 2):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                            (ksize_h, 4),
                                                            (r_h, 2),
                                                            (int(ksize_h / 4.), 1)),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                        (ksize_h, 4),
                                                        (r_h, 2),
                                                        (int(ksize_h / 4.), 1)),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        # 相比上面的CAM，这个在汇总了特征编码的输出之后，应用了一个卷积+sigmoid将输出压缩到1维，
        # 并将shape(B, C, H, W)转化为shape(B, W, C, H)，然后直接执行反卷积，
        x = input[0]
        for i in range(0, len(self.fpn) - 1):
            x = self.fpn[i](x) + input[i + 1]
        # Reducing the input to 1-D form
        x = self.fpn[-1](x)
        # Transpose B-C-H-W to B-W-C-H
        x = x.permute(0, 3, 1, 2).contiguous()

        for i in range(0, len(self.deconvs)):
            x = self.deconvs[i](x)
        return x   # 返回注意力图，shape(N, maxT, H/rh, W/rw)，示例：若输入高宽192,2048，则(nB, 150, 1, 128)


'''
Decoupled Text Decoder。解耦文本解码器：使用注意力图和特征提取图进行最终的识别
'''


class DTD(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout=0.3):
        """
        :param nclass: 类别数（包含未知符号和结束符号）
        :param nchannel: 用于LSTM的输入通道数， 与特征提取阶段最后的输出的通道一致
        :param dropout: dropout比率
        """
        super(DTD, self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True)  # 输入的特征size，隐藏状态的特征size，双向LSTM
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)  # 输入的batch中每一元素的特征size，batch中每一元素的隐藏状态的特征size
        self.generator = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nchannel, nclass)
        )
        self.char_embeddings = Parameter(torch.randn(nclass, nchannel))  # 加到parameter()迭代器中，作为可训练参数。用于表示标签的嵌入向量

    def forward(self, feature, A, text, text_length, test=False):
        """
        :param feature: 特征解码模块最后一层输出的特征图，shape(B, C, H/rh, W/rw)，scene数据集的rh=rw=4, C=512
        :param A: CMA输出，即注意力图，shape(B, maxT, H/4, W/4)，scene数据集的maxT=25
        :param text: 标签
        :param text_length: 各个标签的长度，list
        :param test: 若为True，表示测试集
        :return: 返回识别结果
        """
        nB, nC, nH, nW = feature.size()
        # print("print: ", feature.size())
        nT = A.size()[1]   # 即maxT
        # Normalize。标准化注意力图A，用A的每一项除以所在通道的所有项之和
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)  # view是reshape张量，sum(2)是计算第二维的和
        # weighted sum。执行DTD中的第一步，计算上下文向量Ct
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)  # shape(B, maxT, C, H/4, W/4)

        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)   # shape(maxT, B, C)，对应于(seq_len, batch, input_size)。

        C, _ = self.pre_lstm(C)   # 应用一层LSTM。输入C的shape(maxT, B, C)中C需与nchannel一致，输出C的shape(maxT, B, C)

        C = F.dropout(C, p=0.3, training=self.training)  # dropout，将C中的元素以概率p归零

        if not test:
            lenText = int(text_length.sum())   # batch中标签总长
            nsteps = int(text_length.max())    # 标签最大长度<maxT

            gru_res = torch.zeros(C.size()).type_as(C.data)  # 返回与C同shape(maxT, B, nchannel)，同数据类型的全为0的tensor
            out_res = torch.zeros(lenText, self.nclass).type_as(feature.data)  # shape(lenText, nclass)
            out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)  # shape(lenText, nH, nW)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)   # shape(B, nchannel)
            # 将self.char_embeddings首行重复nB行组成的shape为(nB, nchannel)的tensor。表示上一步解码结果的嵌入向量（初始时）
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            for i in range(0, nsteps):  # 对每一时刻（位置字符）应用GRU
                hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim=1),
                                  hidden)  # 输入shape(B, channel+channel)， 输出隐藏状态shape(B, nchannel)
                gru_res[i, :, :] = hidden   # shape(maxT, B, channel)

                # 上一位置的嵌入向量，（训练时用真实标签的嵌入向量，测试用的是预测标签的嵌入向量）
                prev_emb = self.char_embeddings.index_select(0, text[:, i])  # shape(B, channel)
            gru_res = self.generator(gru_res)  # 对上一步输出应用线性分类，输出shape(maxT, B, nclass)
            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start: start + cur_length] = gru_res[0: cur_length, i, :]
                out_attns[start: start + cur_length] = A[i, 0:cur_length, :, :]
                start += cur_length

            return out_res, out_attns  # 返回预测结果(与真实标签的展平状态的形式一样)和相应的注意力图

        else:
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, nB, self.nclass).type_as(feature.data)  # shape(maxT, B, nclass)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)  # shape(B, nchannel)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))  # shape(B, nchannel)
            out_length = torch.zeros(nB)  # shape(nB)
            now_step = 0
            while 0 in out_length and now_step < nsteps:  # 对每一字符位置，应用GRU，然后线性分类
                hidden = self.rnn(torch.cat((C[now_step, :, :], prev_emb), dim=1), hidden)   # shape(B, nchannel)
                tmp_result = self.generator(hidden)   # shape(B, nclass)
                out_res[now_step] = tmp_result  # 记录预测结果
                tmp_result = tmp_result.topk(1)[1].squeeze()  # 返回tmp_result中每行的最大一个值所在的索引，shape(B)
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1  # 记录batch中每一样本的预测标签长度，‘0’表示标签的结束标志
                prev_emb = self.char_embeddings.index_select(0, tmp_result)  # 上一步的预测结果的嵌入向量
                now_step += 1
            for j in range(0, nB):  # 如果batch中某一样本的预测标签没结束符，就默认其结束符在第nstep-1个位置
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps

            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start: start + cur_length] = out_res[0: cur_length, i, :]
                start += cur_length

            return output, out_length  # 返回预测输出(shape(out_length.sum, nclass))和对应的(各个样本标签)输出长度(shape(nB))
