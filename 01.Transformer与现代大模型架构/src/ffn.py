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
            - Router：对词向量做线性变换再激活，得到不同token对每个路由专家的概率（侧重占比），然后据此决定选择哪些专家————Routing Strategy（路由策略）
            - Experts：多个独立的 FFN。
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
        【优缺点】：
            优点：
                - 稀疏激活，降低计算量
                - 路由可学习，不需人工制定规则
                - 专家具有一定的专业化能力
            缺点：
                - 负载不均衡
                - 路由存在离散选择，无法直接对专家索引进行普通梯度传播，通常是利用专家的路由权重等可微路径训练 Router
                - 落选专家拿不到梯度，无法训练
                - 专家可能出现功能重复：即使有很多专家，也不意味着它们一定会形成不同的专业分工

    3.Shared Expert
        【Top-K 存在的问题】：
            1.共性知识的浪费：不同 token 之间其实存在大量的通用特征提取，如果这些知识也要靠路由专家承载，那每个专家都得"被迫"学一部分通用能力，导致专家间参数冗余，分工不明确
            2.专家冗余：当每个专家都被通用知识"稀释"后，它们学到的功能趋于相似，违背了"专家分工互补"的初衷
        【组成】：
            1.Shared Experts
            2.Router + Routed Experts
        【优缺点】：
            优点：
                - 减少通用特征的重复学习，并分离通用知识和特定知识，有利于分工
                - 保留稳定共性能力：每个 token 都确定性地经过 shared expert，不依赖 router 的判断，通用能力不会被路由波动"丢失"
            缺点：
                - 增加shared expert的计算量
                - 共享专家可能成为计算瓶颈：所有 Token 都必须经过共享专家，在分布式场景下，共享专家可能成为需要重点优化的计算部分
                - 可能抑制路由探索：shared expert 对所有 token 做了统一的变换，会让 router 看到的 token 表征多样性下降，路由决策变得不够"探索"

    4.Fine-grained Expert 细粒度专家
        将原本数量较少、规模较大的 Expert，拆分成数量更多、规模更小的 Expert，让 Router 能够进行更加精细的专家组合
        【传统 MoE 的问题】：
            - 专家粒度不够细：参数量固定时，专家越大，能放下的专家就越少，专家组合的可能性就少，分工粒度粗
            - 专家组合不够灵活：不同 Token 可能只需要某个 Expert 的一部分特征提取能力，但传统 MoE 必须激活整个 Expert。
        【原理】：
            把大专家切成多个小专家，同时成比例增加激活的专家数量，激活参数量基本保持一致（num_expert和d_ff不一定非要同比放大或缩小）。
            假设我们把每个原始 Expert 拆成 4 个更小的 Expert：
                - 原来：8 个 Expert，每个维度 4096，激活 Top-2。
                - 现在：32 个 Expert，每个维度 1024， 激活 Top-8。
            这样每个 token 能从更多的"知识碎片"中自由组合出自己需要的特征提取模式，组合空间指数级增大，专家分工也更细化
        【优缺点】：
            优点：
                - 提高专家组合的灵活性，同时可以组合出更加丰富的特征提取方式，知识组合更加精细
                - 同等参数/计算下效果更强
            缺点：
                - 路由和调度开销增加：router 需要探索的组合空间指数级变大
                - 专家池指数级变大，训练和负载均衡也会更加复杂
                - 单个专家的表达能力可能下降
                - 专家太小会伤 GPU 效率：单个专家矩阵变小后，矩阵乘法的 GPU kernel 利用率下降，碎成太多小矩阵反而变慢

    5.代表组合大致这几种，万变不离其宗，只是参数不一样
        5.1.Mixtral MoE
            Top-K Routing + SwiGLU Experts
        5.2.DeepSeek MoE
            SwiGLU Experts (Shared Expert + Fine-grained ) + Router 概率用 Sigmoid
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
        topk_logits, topk_ids = torch.topk(logits, k=self.top_k, dim=-1)

        # 计算权重: [batch_size, seq_len, top_k]
        topk_weights = topk_logits / topk_logits.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids


class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()

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

        # 展开token:[batch_size * seq_len, d_model]
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


class SharedExpertMoE(nn.Module):
    '''
    改变的地方：
    1.__init__多一个初始化
    2.forward 有共享专家累加，路由部分一模一样
    '''
    def __init__(self, d_model, d_ff, num_experts, top_k, num_shared_experts=1):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_shared_experts)])

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, d_model]
        '''
        batch_size, seq_len, d_model = x.shape

        # 展开token:[batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)


        # --------- 1.Routed Experts ---------
        # Routing路由，并展开: [batch_size * seq_len, top_k]
        weights, ids = self.router(x)
        weights_flat = weights.view(-1, self.top_k)
        ids_flat = ids.view(-1, self.top_k)

        # 初始化Routed Experts输出：[batch_size * seq_len, d_model]
        routed_output = torch.zeros_like(x_flat)
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
            routed_output.index_add_(dim=0, index=token_idx, source=expert_output * expert_weights)


        # --------- 2.Shared Experts ---------
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output += expert(x_flat)


        # --------- 3.整合输出 ---------
        output = routed_output + shared_output
        output = output.view(batch_size, seq_len, d_model)

        return output


class FinegrainedMoE(nn.Module):
    '''
    和 TopKMoE 基本一致，只是 d_ff, num_experts, top_k 不一样
    '''
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.moe = TopKMoE(d_model, d_ff, num_experts, top_k)

    def forward(self, x):
        return self.moe(x)


class DeepSeekRouter(nn.Module):
    '''
    不同：
    1.sigmoid
    2.route_scale

    1.为什么要用 sigmoid？
    softmax 下专家分数互相耦合（抢同一份概率质量），这会存在2个问题：
        - 赢家通吃：训练早期，某个专家偶然获得稍高的 logit，softmax 会把它放大成显著的概率优势，进而形成正反馈
        - 耦合导致专家冗余：Router 学到的决策边界本质是专家之间的相对比较。想把某个专家往某个方向推，会同时改变其他所有专家的相对位置。专家难以独立地探索自己的分工方向，最终学得都差不多，违背“分工互补”的初衷
    而使用 sigmoid 每个专家的分数只依赖自己的 logit，与其他专家无关

    2.为什么要乘以 route_scale？
    在由 topk_logits 归一化计算 topk_weights时，routed experts 的权重和强制等于1，输出的"总能量"被硬性钉死在"routed experts输出的加权和 ≈ 1 份"，对比如下：
        - shared expert：每个 token 确定性通过，输出不做任何缩放
        - routed experts：输出是归一化权重的加权和，而且权重和为1
    这就导致 routed experts 和 shared experts 输出的量级失衡：routed experts 被系统性压小 → 量级失衡 → 学习信号失衡。
    而且 Fine-grained 放大了这个问题。专家切得越小（d_ff 从 4096 → 1024），单个专家的输出能量本来就小；再叠加归一化，routed 部分就几乎"听不见"了
    DeepSeek-V3 取 routed_scale = 2.5，本质上是在做量级匹配
    '''
    def __init__(self, d_model, num_experts, top_k, route_scale=2.5):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale

        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        return:
            topk 权重：[batch_size, seq_len, top_k]
            topk 索引：[batch_size, seq_len, top_k]
        '''
        # 每个 token 对应的专家概率：[batch_size, seq_len, num_experts]
        logits = self.router(x)     
        logits = torch.sigmoid(logits, dim=-1)

        # top-k: [batch_size, seq_len, top_k]
        topk_logits, topk_ids = torch.topk(logits, k=self.top_k, dim=-1)

        # 计算权重: [batch_size, seq_len, top_k]
        topk_weights = topk_logits / topk_logits.sum(dim=-1, keepdim=True)

        # 路由权重缩放
        topk_weights = topk_weights * self.route_scale

        return topk_weights, topk_ids


# todo Load Balancing Loss负载均衡损失
# todo Auxiliary Loss
# todo Router Z-Loss
# todo Expert Capacity
# todo Efficient Dispatch高效分发token
class DeepSeekMoE(nn.Module):
    '''
    不同：
    1.router 用 sigmoid + route_scale
    2.expert 用 SiLU
    3.forward 和 SharedExpertMoE 一模一样
    '''
    def __init__(self, d_model, d_ff, num_experts, num_shared_experts, top_k, route_scale=2.5) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts= num_shared_experts

        self.router = DeepSeekRouter(d_model, num_experts, top_k, route_scale)
        self.experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_shared_experts)])

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len,d_model]
        '''
        batch_size, seq_len, d_model = x.shape

        # 展开token:[batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # --------- 1.Routed Experts ---------
        # Routing路由，并展开: [batch_size * seq_len, top_k]
        weights, ids = self.router(x)
        weights_flat = weights.view(-1, self.top_k)
        ids_flat = ids.view(-1, self.top_k)

        # 初始化Routed Experts输出：[batch_size * seq_len, d_model]
        routed_output = torch.zeros_like(x_flat)
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
            routed_output.index_add_(dim=0, index=token_idx, source=expert_output * expert_weights)

        # --------- 2.Shared Experts ---------
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output += expert(x_flat)


        # --------- 3.整合输出 ---------
        output = routed_output + shared_output
        output = output.view(batch_size, seq_len, d_model)

        return output