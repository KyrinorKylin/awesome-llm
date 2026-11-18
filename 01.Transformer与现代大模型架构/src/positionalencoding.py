'''
1.绝对位置编码：
    - Sinusoidal PE
    - Learnable PE（可学习位置编码）
        - 和嵌入基本一致，只不过它们是拿[batch_size, seq_len]初始化，这里是拿[total_max_len]初始化
        - 缺点：如果输入过长，可能句子末尾那部分的位置编码训练较少；超出长度甚至没有位置编码

2.相对位置编码：
    - T5 Relative Position Bias
        - 在qk计算完了之后再加一个可学习的位置编码矩阵B（偏置），Bij = 第i个token的索引 - 第j个token的位置索引，以此表示相对位置，然后分桶得到B。
        - 缺点：远距离分辨率下降，多个距离共用一个 Bucket

3.旋转位置编码：
    - RoPE———当下的核心
        作用于 QK 上
        - 对于每一个token的嵌入向量，两个维度为一组，左乘二维旋转矩阵，这样到时候计算QK时由线性代数知QK之间还乘了一个相对位置的旋转矩阵。
        - 也就是说，一开始根据绝对位置旋转，但计算QK时变成了相对位置。
        - 缺点：
            1.训练时如果只见过较短的序列，推理时直接扩展到很长的序列，模型可能无法适应新的旋转角度分布

4.基于Attention Bias的位置偏置：
    - ALiBi
        - 在计算完 qk 之后，加上偏置惩罚矩阵B，Bij=(i-j)*m_h，m_h=2**(注意力头的编号)。为不同的 Attention Head 设置不同的固定斜率，让不同的头关注不同范围的上下文

5.RoPE 的长上下文扩展变体：
    - YaRN
'''
import math
import torch
from torch import nn


'''正弦位置编码'''
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, total_max_len=2000):
        '''
        total_max_len: 是全局静态上限，要大于数据集中所有句子的长度, 而 seq_len 只是每个batch中的最大长度
        '''
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.total_max_len=total_max_len

        # 1.定义位置编码向量，先初始化框架，值后续填充
        pe=torch.zeros(total_max_len, d_model)

        # 2.定义列向量，表明句子中所有token的位置：[total_max_len，1]
        position=torch.arange(0, total_max_len, dtype=torch.float).unsqueeze(1)

        # 3.定义变化矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 4.计算三角函数里面的值
        position_val=position * div_term

        # 5.填充位置编码
        pe[:, 0::2] = torch.sin(position_val)
        pe[:, 1::2] = torch.cos(position_val)

        # 6 升维，方便对 batch 维广播: [total_max_len, d_model] -> [1, total_max_len, d_model]
        pe=pe.unsqueeze(0)

        # 7 注册为buffer: 可以调用self.pe，还能额外声明不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, d_model]
        '''
        seq_len = x.shape[1]
        if seq_len > self.total_max_len:
            raise ValueError(
                f"输入序列长度 ({seq_len}) 超过了位置编码最大支持长度 ({self.total_max_len})！请增大 total_max_len 参数。")

        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


'''可学习位置编码'''
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, total_max_len=2000):
        super().__init__()
        self.total_max_len=total_max_len

        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(total_max_len, d_model)

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, d_model]
        '''
        seq_len = x.shape[1]
        if seq_len > self.total_max_len:
            raise ValueError(f'句子长度太长了！超出了total_max_len={self.total_max_len}')

        # 先初始化为绝对位置[seq_len]，再嵌入为[seq_len, d_model]
        position = torch.arange(seq_len)
        position = self.pe(position)

        return self.dropout(x + position)


'''旋转位置编码'''
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k, max_seq_len=2000, base=10000.0):
        super().__init__()
        assert d_k//2 ==0, 'd_k不为偶数，RoPE无法进行！'
        self.d_k = d_k

        # 1.获取计算参数：token索引i和第k对维度
        i = torch.arange(max_seq_len)
        k = torch.arange(0, d_k, 2)

        # 2.计算频率w:[d_k//2]
        w = base ** (-k / d_k)

        # 3.所有token的旋转角度: [max_seq_len, d_k//2]
        angle = i.unsqueeze(1) * w.unsqueeze(0)

        self.register_buffer('cos_angle', torch.cos(angle))
        self.register_buffer('sin_angle', torch.sin(angle))

    def forward(self, x, token_ids):
        '''
        :param x: [..., seq_len, d_k]
        :param token_ids: [..., seq_len]储存每个token的位置下标
        :return: 和x一致，[..., seq_len, d_k]
        '''
        # 不是切片，而是根据token的位置索引取值：[..., seq_len, d_k//2] 。为什么不直接[:seq_len]？因为索引不一定是0-base，而且甚至不一定连续！
        cos = self.cos_angle[token_ids]
        sin = self.sin_angle[token_ids]

        # 取每一对维度的第一个和第二个
        x_first = x[..., 0::2]  #不管前面有多少维，只对最后一维切片
        x_second = x[..., 1::2]

        # 初始化变换后的矩阵，值之后再填充
        out = torch.empty_like(x)

        # 填充变化后的值
        out[..., 0::2] = x_first * cos - x_second * sin   # *是对应元素相乘
        out[..., 1::2] = x_first * sin + x_second * cos

        return out


'''ALiBi'''
class ALiBi(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        # 计算不同注意力头的斜率
        m_h = self.get_m_h(num_heads)

        self.register_buffer('m_h', m_h)

    @staticmethod
    def get_m_h(num_heads):
        # 假设num_heads是2的幂
        start = 2 ** (-8 / num_heads)

        m_h = torch.tensor([start ** (i+1) for i in range(num_heads)])

        return m_h

    def forward(self, seq_len):
        '''
        :param seq_len:
        :return: 偏置惩罚矩阵[1, head, seq_len, seq_len]
        '''
        # 每个token的位置: [seq_len]
        token_pos = torch.arange(seq_len)

        # token之间的相对位置: [seq_len, seq_len]
        distance = token_pos[:, None] - token_pos[None, :]  # None和unsqueeze等价，只是写起来方便

        # 偏置惩罚: [head, seq_len, seq_len]
        bias = -self.m_h[:, None, None] * distance

        # [1, head, seq_len, seq_len]
        return bias.unsqueeze(0)