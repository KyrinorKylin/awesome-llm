'''
一、激活函数变体
    1.ReLU
        原始实现
    2.GELU
    3.SiLU / Swish
    4.Squared ReLU

二、门控 FFN
    1.GLU（Gated Linear Unit）
        激活函数的不同衍生出了2和3
        【原理】：
            输入经过两条不同的线性变换，一条负责生成特征，另一条负责生成门控信号，两者逐元素相乘
            - 特征分支：只升维处理：负责提取和生成特征，不需要提前限制数值范围，可以产生任意正数或负数，所以不需要激活函数
            - 门控分支：升维 + 激活：负责判断特征应该如何通过，同一个特征，不同token可以根据门控分支的输出，动态决定保留多少
        【Q&A】
        1.两条路为什么比一条路更有表达能力？
            对比普通 FFN 和 GLU 的数据传递过程：
                普通 FFN: h=f(xW_1)
                GLU:      h=f(xW_1) * (xW_2)    W1是门控分支，W2是特征分支
            GLU 的每一个中间特征实际上由两个不同的可学习投影共同决定，一个负责生成数值，一个负责调节数值，这种乘法交互能够引入更丰富的非线性表达能力
            可以理解为，普通 FFN：学习如何变换特征；GLU：学习如何生成特征，以及在什么条件下让这些特征发挥作用

    2.SwiGLU———现代主流
        激活函数用SiLU

    3.GEGLU
        激活函数用GELU

三、MoE FFN
    1.传统 MoE（了解传统 MoE 的构成）
        【为什么要MoE】？
            原始结构所有 Token 都共享相同的 FFN 参数，但不同 Token 可能需要不同的特征提取能力，那么能不能让不同 Token 选择不同的 FFN 呢？
        【基本结构】：
            - Router（路由器）：对词向量做线性变换再激活，得到不同token对每个路由专家的概率（侧重占比），然后据此决定选择哪些专家————Routing Strategy（路由策略）
            - Experts（专家网络）：多个独立的 FFN。
        【Routing Strategy（路由策略）有哪些】？
            Top-1、Top-2、Top-K
        【负载均衡（Load Balancing）】：
            训练时，不同专家的训练程度可能不同：
                - 部分专家接收大量 Token，产生计算瓶颈。
                - 部分专家几乎没有 Token，无法充分训练。
            相关技术：
                - Auxiliary Load Balancing Loss
                - Router Z-Loss
                - Expert Choice Routing
                - Expert Capacity
        【专家分工（Expert Specialization）】：
            假设有 8 个 Expert，如果它们最终学习到的参数和功能都非常相似，那么增加 8 个专家就没有发挥应有的价值
            我们希望：
                - 一些专家学习某些类型的特征。
                - 另一些专家学习不同类型的特征。
                - 专家之间形成互补，而不是大量重复
            相关技术和研究方向：
                - Shared Experts
                - Fine-grained Experts
                - Expert Specialization

    2.Top-K Routing
    3.Shared Expert
    4.Fine-grained Expert
    5.代表组合
        5.1.Mixtral MoE
        5.2.DeepSeek MoE
'''
# todo Router能不能像注意力机制那样计算概率
from turtle import forward
from typing import Literal
import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation: Literal['relu', 'gelu', 'silu']):
        super().__init__()
        activation_fn={
            'relu': nn.ReLU(), 
            'gelu': nn.GELU(), 
            'silu': nn.SiLU()
        }

        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = activation_fn[activation]
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class GLU(nn.Module):
    def __init__(self, d_model, d_ff, activation: Literal['gelu', 'silu']):
        super().__init__()
        activation_fn={
            'gelu': nn.GELU(), 
            'silu': nn.SiLU()
        }

        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

        self.act = activation_fn[activation]

    def forward(self, x):
        gate = self.act(self.gate(x))
        up = self.up(x)

        x = gate * up
        return self.down(x)


# todo 负载均衡损失
# todo Expert Parallelism
# todo Auxiliary Loss
# todo Expert Capacity
# todo 高效分发
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        '''
        TopKRouter为什么设置bias=False？
            1.Top-K 只关心相对排序，bias 提供的信息是冗余的
            2.bias 会与负载均衡损失“打架”: 如果 Gate 带 bias，训练早期可能让某个专家获得一个正的 b，导致所有 token 都系统性偏向它，形成“赢家通吃”的先验
        '''
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # 每个token的词向量需要哪些专家，输入维度是d_model
        self.router = nn.Linear(d_model, num_experts, bias=False)   #bias=False减少参数量，

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        return:
            topk 权重：[batch_size, seq_len, top_k]
            topk 索引：[batch_size, seq_len, top_k]
        '''
        # 每个 token 对应的专家概率：[batch_size, seq_len, num_experts]
        logits = self.router(x)
        logits = torch.softmax(logits, dim = -1)

        # top-k: [batch_size, seq_len, top_k]
        topk_logits, topk_ids = torch.topk(logits, dim=-1)

        # 计算权重: [batch_size, seq_len, top_k]
        topk_weights = topk_logits / topk_logits.sum(dim=-1, keep_dim=True)

        return topk_weights, topk_ids


class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super()._init_()

        self.num_experts =num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        '''
        这里有一个展开操作，把 batch 中所有 token 拼在一起得到[batch_size * seq_len, top_k]，然后有相同专家的 token 就一起经过 expert 处理
        :param x: [batch_size, seq_len, d_model]
        :return: 最终每个词向量都是该词向量经过不同专家后的词向量的加权和
        '''
        batch_size, seq_len, d_model = x.shape

        # 向量展开:[batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # Routing路由，并展开: [batch_size * seq_len, top_k]
        weights, ids = self.router(x)
        weights_flat = weights.view(-1, self.top_k)
        ids_flat = ids.view(-1, self.top_k)

        # 初始化输出：[batch_size * seq_len, d_model]
        output = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            # token_idx：满足条件的行坐标，即选择了该专家的 token id；topk_idx：满足条件的列坐标，即对于每个 token，该专家是 top几
            # [n], [n]：n 是不定量，表示当前专家接收多少 token
            token_idx, topk_idx = torch.where(ids_flat == expert_idx)

            # 不管是多少维的向量，都能获取原始个数：没有任何 token 选择当前 expert，跳过
            if token_idx.numel() == 0:
                continue

            # 选择该专家的 token 及向量：[n, d_model]
            expert_input = x_flat[token_idx]

            # [n, d_model]
            expert_output = expert(expert_input)

            # 获取不同 token 在该专家上的权重：unsqueeze 前是 [n]，unsqueeze 后是 [n, 1]
            expert_weights = weights_flat[token_idx, topk_idx].unsqueeze(-1)

            # 按照指定维度和索引填充值：沿着第 0 维累加；* 是逐元素相乘，有广播效应
            output.index_add_(dim=0, index=token_idx, source=expert_output * expert_weights)

        output = output.view(batch_size, seq_len, d_model)

        return output