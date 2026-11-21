'''
按"解决什么问题"分类的常见变体

一、注意力头结构优化———降低推理显存与带宽：KV Cache 相关（LLM 推理最核心）
自回归生成时要缓存历史的 K、V，长上下文下 KV Cache 常常比模型权重还大，也是解码阶段的带宽瓶颈
    1.MHA
    2.MQA（Multi-Query Attention）
        【原理】：多个 Query Head，共享一组 Key 和 Value.
        【优点】：
            - 大幅减少 KV Cache。
            - 减少自回归生成时 KV 的读取量。
            - 提升推理阶段的内存带宽利用效率
        【缺点】：
            - 所有 Query Head 共享 K、V，可能降低模型的表达能力
    3.GQA（Grouped-Query Attention）
        GQA 可以看作 MHA 和 MQA 的折中方案：将 Query Head 分成若干组，每组共享一组 K、V。
    4.MLA（Multi-Head Latent Attention）
        【思想】：进一步压缩缓存，不直接缓存完整的 K、V，而是缓存压缩后的潜在向量（Latent）
        【流程】（训练阶段无需 KV Cache，其他的都一样）：
            1.获取每一个transformer层的输入向量（也就是上一层的输出或词嵌入向量）h；
            2.然后读取历史缓存 c_kv_old（压缩后的潜在向量）,k_r_old（用于位置编码，没有压缩，所以不需要升维）；
            3.对 q 降维得到 c_q，再升维恢复为原始形状 q_c；
            4.再对 q 进行RoPE编码：对 c_q 额外再升维，然后对其每个头应用RoPE得到 q_r；
            5.计算上一轮刚预测的新token的KV Latent：c_kv_new = 降维矩阵 * h，
            6.计算新token的RoPE：对 h 乘以一个投影矩阵，再应用RoPE得到 k_r_new；
            7.缓存与新计算的合并，再存入缓存：合并 c_kv_old,c_kv_new 得到最终 c_kv，合并 k_r_old,k_r_new 得到最终 k_r。最后把这些存入缓存；
            8.对完整的 c_kv 分别乘以两个升维矩阵得到 k_c,v_c；
            9.合并 k_c,k_r 得到 k，合并 q_c,q_r 得到 q，v就直接取 v_c
            10.最后就是常规的注意力计算
        特别值得注意的是：可以通过矩阵的乘法结合律进行吸收（Absorption），不必每次都显式恢复完整的 K、V，而是代入升维降维矩阵，先进行特定矩阵的运算可以减小计算量———看公式你就知道了
        【Q&A】
            1.为什么要压缩Q？
                理论上可以不压缩Q，这里是为了进一步优化注意力投影部分的参数和计算开销，降低显存
                - 相当于把q的投影矩阵低秩分解了，减小参数量
                - 减小投影q的矩阵计算量：低秩分解可以减少+和*的次数
            2.为什么计算RoPE要解耦？
                如果直接对 qk 乘以旋转矩阵，代入发现注意力分数 = c_Q,t^T · W_UQ^T · R_{j-t} · W_UK · c_j（j表示第j个token）。旋转矩阵夹在中间，且每一对 (t, j) 的矩阵都不一样，导致：
                - R 和 W_UK 不能互换位置，所以也没法把 W_UK 单独挪到 query 一侧，享受不到吸收的好处
                - 如果缓存旋转后的完整 K，缓存退化回 MHA 量级，MLA 白做了
        【观察】
            1.训练时，从__init__和forward中的代码上看，qkv都是先降维再升维代替了原来的直接投影，顺便产生 *_r 位置编码矩阵
        【优点】：
            - 显著降低 KV Cache，更适合长上下文推理
            - 保留多头注意力的表达能力
            - 通过低秩压缩，在缓存效率与表达能力之间取得平衡
            - 降低计算量
        【缺点】：
            - 结构更复杂。
            - 需要额外的低秩投影与计算。
            - 实现时还要处理 RoPE 与解耦位置编码等问题

二、注意力计算范围/连接模式优化
    1.Full Attention（全注意力）
        标准 Transformer 的基础注意力机制
    2.Sparse Attention（稀疏注意力）
        【为什么要稀疏】
            传统注意力的计算复杂度极大，Sparse Attention认为：既然没有必要让每个 Token 都关注所有 Token，那就减少注意力连接数量
        【原理】
            计算完 QK 之后，加上稀疏掩码 M，由 0 和 -∞ 组成。
        【变体】：
            究竟应该让一个 Token 关注哪些 Token？这就催生了下面 3~6 的变体
    3.Local Attention（局部注意力）
        这是一个大类，属于Sparse Attention，但又包含Sliding Window Attention（最经典的实现方式）
    4.Sliding Window Attention（滑动窗口注意力）
        每个 Q 只关注以它为中心的固定个数的token
        【优点】：
            - 实现简单。
            - 能够充分捕获相邻 Token 的关系。
            - 计算复杂度低。
        【缺点】：
            - 无法直接关注距离较远的 Token。
            - 长距离信息需要经过多层 Transformer 才能逐渐传递
    5.Dilated Local Attention（膨胀局部注意力）
        对 4 的修改，关注的是个等差数列，借鉴了 CNN 中的 Dilated Convolution（空洞卷积）思想
        【原理】：
            第 i 个token关注的位置为: N(i) = {i + kd, k = -r, ..., r}。d 为膨胀系数（步长），r 为窗口大小
            Eg：[T1, T2, ..., T9], T5会关注 T1 T3 T5 T7 T9
        【优点】：
            - 不明显增加计算量。
            - 能够扩大感受野。
        【缺点】：
            - 会跳过中间 Token。
            - 可能遗漏重要的局部关联
    6.Block Sparse Attention（块稀疏注意力）
        不再以单个 Token 为单位决定是否计算注意力，而是以一组 Token 组成的 Block 为单位，计算 Block 之间的注意力。这种设计不仅可以减少计算量，还能比普通Sparse更好地利用 GPU 的并行计算能力
        【分类】：
            - Block Sliding Window（块滑动窗口）
            - Block Random Attention（块随机注意力）：随机选择 Block 进行连接
            - Block Global Attention（块全局注意力）：将部分Block设定为全局Block，需要和所有Block交互
            - Hybrid Block Sparse Attention（混合块稀疏注意力）：综合上述3种方式
    7.Global Attention
        【原理】
            选择特定 token 关注整个序列，其他 token 只关注窗口
    8.DeepSeek Sparse Attention
        不再让每个 Token 固定关注附近的 Token，而是通过一个轻量级的索引器（Lightning Indexer），动态找出当前 Token 最值得关注的历史 Token
        【原理】
            1.Lightning Indexer计算当前 Q 和历史 K 之间的相似度
            2.从所有历史 Token 中挑选 Top-K
            3.MLA 计算注意力分数

三、计算效率与硬件访存优化
    1.FlashAttention（v1/v2/v3）
        现代模型训练的主流
    2.PagedAttention（vLLM）
    3.Ring Attention

四、KV Cache 与推理内存管理
    1.PagedAttention
    2.vAttention
    3.KV Cache Quantization
    4.KV Cache Offloading
    5.Prefix Caching
    
五、注意力分数与权重计算公式
    1.Linear Attention（线性注意力）
    2.Additive Attention（加性注意力）
    3.Cosine Attention、
    4.Performer
    5.RetNet（Retention Network）
'''
import copy
import math
import torch
from torch import nn
from positionalencoding import RotaryPositionalEmbedding


# todo 添加 KV Cache 优化
# todo 添加 RoPE 位置编码
class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])    #Q,K,V的投影矩阵 W
        self.dropout = nn.Dropout(dropout)

        # 变量记录保存注意力分数
        self.atten_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        q,k,v: [batch_size, head, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：[batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]

        注意力计算为什么要除以 math.sqrt(self.d_k)？
        1.点积的方差会随 d_k 增大
            假设 q 和 k 的每个分量相互独立，均值为 0，方差为 1（LayerNorm 和合理的初始化大致能保证这一点）。
            每一项 q_i k_i 的均值为 0、方差为 1，所以整个点积的均值为 0、方差为 d_k，标准差为 sqrt(self.d_k)。
            除以 sqrt(self.d_k) 之后，方差重新回到 1，分数的尺度和 d_k 无关。
        2.分数太大会让 softmax 饱和
        3.为什么是 √d_k，而不是 d_k 或别的？
            目标是把方差从 d_k 压回 1。方差缩放要除以标准差，也就是除以 √d_k。
            除以 d_k 会过度缩小，使分数方差变成 1/d_k，softmax 输出会趋于均匀分布，注意力又失去区分度。
        '''
        # 注意力分数：[batch_size, head, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        # 防不胜防的失活
        if dropout is not None:
            atten_weights = dropout(atten_weights)

        # 加权求和：[batch_size, head, seq_len, d_k]
        attentions = torch.matmul(atten_weights, v)

        #
        return attentions, atten_weights

    def forward(self, q, k, v, mask=None):
        '''
        q,k,v: [batch_size, seq_len, d_model]
        return: 添加注意力后的所有句子向量：[batch_size, seq_len, d_model]
        '''
        batch_size = q.shape[0]

        # q,k,v投影: [batch_size, seq_len, d_model]
        q, k, v = [linear(x) for linear, x in zip(self.linears, (q, k, v))]

        # q,k,v分头
        q, k, v = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) for x in [q, k, v]]

        # 注意力计算: [batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        attens, self.atten_weights = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 结果融合: [batch_size, seq_len, d_model]
        atten_cat = attens.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 还有一个线性层
        return self.linears[-1](atten_cat)


# todo 添加 KV Cache 优化
# todo 添加 RoPE 位置编码
class MQA(nn.Module):
    '''
    和 MHA 不一样的是：k,v的投影矩阵、不用分头而是unsqueeze
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k) #和MHA有一点点不一样，因为MQA的KV只有一个头，所以这里直接映射了。如果还和W_q一样后面用view，那就失去了MQA的意义了。
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        q: [batch_size, head, seq_len, d_k]
        k,v: [batch_size, 1, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：[batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        '''
        # 注意力分数：[batch_size, head, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        # 防不胜防的失活
        if dropout is not None:
            atten_weights = dropout(atten_weights)

        # 加权求和：[batch_size, head, seq_len, d_k]
        attentions = torch.matmul(atten_weights, v)

        #
        return attentions, atten_weights

    def forward(self, q, k, v, mask=None):
        '''
        q,k,v: [batch_size, seq_len, d_model]
        return: 添加注意力后的所有句子向量：[batch_size, seq_len, d_model]
        '''
        batch_size = q.shape[0]

        # q,k,v投影：[batch_size, seq_len, d_model]，[batch_size, seq_len, d_k]，[batch_size, seq_len, d_k]
        q, k, v = [linear(x) for linear, x in zip([self.W_q, self.W_k, self.W_v], (q, k, v))]

        # q分头：[batch_size, seq_len, num_heads, d_k], [batch_size, 1, num_heads, d_k]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 注意力计算: [batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        attens, self.atten_weights = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 结果融合
        atten_cat = attens.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(atten_cat)


# todo 添加 KV Cache 优化
# todo 添加 RoPE 位置编码
class GQA(nn.Module):
    '''
    和 MHA 不一样的是：kv 的投影矩阵、kv 分头、分组与 q 对齐
    '''
    def __init__(self, d_model: int, num_heads: int, num_groups: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        q,k,v: [batch_size, head, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：[batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        '''
        # 注意力分数：[batch_size, head, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        # 防不胜防的失活
        if dropout is not None:
            atten_weights = dropout(atten_weights)

        # 加权求和：[batch_size, head, seq_len, d_k]
        attentions = torch.matmul(atten_weights, v)

        #
        return attentions, atten_weights

    # todo 五维分组代替repeat_interleave
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # q,k,v投影：[batch_size, seq_len, d_model]，[batch_size, seq_len, num_groups * self.d_k]，[batch_size, seq_len, num_groups * self.d_k]
        q, k, v = [linear(x) for linear, x in zip([self.W_q, self.W_k, self.W_v], (q, k, v))]

        # q,k,v分头：[batch_size, num_heads, seq_len, d_k]，[batch_size, num_groups, seq_len, d_k]，[batch_size, num_groups, seq_len, d_k]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, -1, self.num_groups, self.d_k).transpose(1,2)
        v = v.view(batch_size, -1, self.num_groups, self.d_k).transpose(1,2)

        # 每个group的 kv 分给对应的 q：0~heads_per_group-1 号的 q 用第 0 组 kv，依此类推。
        # [batch_size, num_groups, seq_len, d_k] -> [batch_size, num_heads, seq_len, d_k]
        k = k.repeat_interleave(self.heads_per_group, dim=1)    #But这里用的是复制，增加了显存占用，失去了GQA的优势，应该用五维分组
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        # 注意力计算: [batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        attens, self.atten_weights = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 结果融合
        atten_cat = attens.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(atten_cat)


# todo 添加 KV Cache 优化
# todo 添加矩阵吸收
class MLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_c_kv: int, d_c_q: int, d_rope: int, dropout: float, max_seq_len: int = 2000):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头里不带 RoPE 的维度，v 的维度也用它
        self.d_c_kv = d_c_kv             # KV latent 维度，一般远小于 d_model
        self.d_c_q = d_c_q               # Q latent 维度
        self.d_rope = d_rope             # 解耦 RoPE 的维度

        # Q：下投影（压缩）-> 上投影（还原）。q_r 从 c_q 来，每个头一份
        self.W_DQ = nn.Linear(d_model, d_c_q)
        self.W_UQ = nn.Linear(d_c_q, d_model)
        self.W_QR = nn.Linear(d_c_q, num_heads * d_rope)

        # KV：W_DKV 压缩出 latent，K、V 共用这一个 latent，各自一个上投影
        self.W_DKV = nn.Linear(d_model, d_c_kv)
        self.W_UK = nn.Linear(d_c_kv, d_model)
        self.W_UV = nn.Linear(d_c_kv, d_model)
        self.W_KR = nn.Linear(d_model, d_rope)

        self.W_O = nn.Linear(d_model, d_model)

        # RoPE 只作用在 d_rope 这一小段上，所以传进去的是 d_rope 而不是 d_k
        self.rope = RotaryPositionalEmbedding(d_rope, max_seq_len)

        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q_c, q_r, k_c, k_r, v, mask, dropout: nn.Dropout):
        '''
        q_c: [batch_size, head, seq_len, d_k]
        q_r: [batch_size, head, seq_len, d_rope]
        k_c: [batch_size, head, seq_len, d_k]
        k_r: [batch_size, 1, seq_len, d_rope]
        v:   [batch_size, head, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：[batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]

        q = [q_c ; q_r]，k = [k_c ; k_r] 拼起来做点积，等价于两部分分别点积再相加。
        这里用相加的写法，k_r 就不用 expand 成 num_heads 份再 cat，直接靠广播（和 MQA 一样）。
        缩放系数也要跟着变：点积的维度是 d_k + d_rope，所以除以 sqrt(d_k + d_rope)，而不是 sqrt(d_k)。
        '''
        # 注意力分数：[batch_size, head, seq_len, seq_len]
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) + torch.matmul(q_r, k_r.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k + self.d_rope)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        # 防不胜防的失活
        if dropout is not None:
            atten_weights = dropout(atten_weights)

        # 加权求和：[batch_size, head, seq_len, d_k]
        attentions = torch.matmul(atten_weights, v)

        return attentions, atten_weights

    def forward(self, x, token_ids=None, mask=None):
        '''
        x: [batch_size, seq_len, d_model]
        token_ids: [seq_len]，每个 token 的位置下标，不传就默认 0 ~ seq_len-1
        return: 添加注意力后的所有句子向量：[batch_size, seq_len, d_model]

        注意：K、V 来自同一个 latent，所以 MLA 只能接一个输入 x（自注意力），不像 MHA/MQA/GQA 那样 q,k,v 分开传。
        '''
        batch_size, seq_len = x.shape[0], x.shape[1]
        if token_ids is None:
            token_ids = torch.arange(seq_len, device=x.device)

        # Q：压缩 -> 上投影：[batch_size, seq_len, d_c_q]，[batch_size, seq_len,d_model]，[batch_size, seq_len, num_heads * d_rope]
        c_q = self.W_DQ(x)
        q_c = self.W_UQ(c_q)
        q_r = self.W_QR(c_q)

        # KV：压缩出 latent（KV Cache 存的就是它）：[batch_size, seq_len, d_c_kv]
        c_kv = self.W_DKV(x)
        # latent 上投影还原出每个头的 k、v：[batch_size, seq_len, d_model]
        k_c = self.W_UK(c_kv)
        v = self.W_UV(c_kv)
        # 解耦的 key：[batch_size, seq_len, d_rope]
        k_r = self.W_KR(x)

        # 分头：[batch_size, num_heads, seq_len, d_k]，[batch_size, num_heads, seq_len, d_rope]
        q_c = q_c.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q_r = q_r.view(batch_size, -1, self.num_heads, self.d_rope).transpose(1, 2)
        k_c = k_c.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # k_r 只有一个头，不分头而是 unsqueeze：[batch_size, 1, seq_len, d_rope]
        k_r = k_r.unsqueeze(1)

        # 只有 r 的那部分做 RoPE。token_ids: [seq_len] -> cos/sin: [seq_len, d_rope//2]，自动广播到 batch 和 head 维
        q_r = self.rope(q_r, token_ids)
        k_r = self.rope(k_r, token_ids)

        # 注意力计算: [batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        attens, self.atten_weights = self.scaled_dot_product_attention(q_c, q_r, k_c, k_r, v, mask, self.dropout)

        # 结果融合
        atten_cat = attens.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 注意：__init__ 里定义的是 self.W_O（大写O），这里保持一致
        return self.W_O(atten_cat)