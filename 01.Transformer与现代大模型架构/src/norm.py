'''
一、LayerNorm
    【原理】：
        每个词向量减去均值，再除以标准差———同时进行均值中心化和方差归一化
二、RMSNorm
    【原理】：
        RMSNorm 认为，LayerNorm 的均值中心化操作并不是必需的，因此直接除以均方根（RMS），再乘以缩放系数 γ
        注：γ 是可学习的向量，维度和词向量一样
    【为什么要乘以缩放系数】？
        纯粹的归一化会限制网络的表达能力，输出被强制约束在一个固定的尺度上，网络失去了对激活值“大小”的控制权
        具体来说：
            - 网络可能需要不同的特征有不同的强度。例如，某些特征信号本身就应该“强”（幅度大），某些特征就应该“弱”（幅度小）。归一化把所有向量的 RMS 都拉到 1，等于强行抹平了这种天然的幅度差异。
            - 归一化操作是“不可逆的信息破坏”：一旦所有向量都被标准化到同一尺度，网络在后续层就再也无法区分“原本就大的值”和“原本就小的值”了
    【为什么缩放系数不是一个固定数值】？
        因为一个词向量的不同维度可能蕴含不同的知识，需要不同的数值尺度。而归一化把所有值压缩到同一尺度，于是配置可学习参数，让优化过程自己找到最优解

对比	                LayerNorm	    RMSNorm
均值中心化	                有	            无
方差计算	                有	            无
RMS 计算	                无	            有
可学习缩放 \(\gamma\)	    有	             有
可学习偏置 \(\beta\)	   通常有	    原始版本没有
计算开销	              相对较高	        相对较低
'''
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps

        # 可学习的缩放和偏置参数
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x * self.weight + self.bias

        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # 可学习的缩放系数: [d_model]
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight