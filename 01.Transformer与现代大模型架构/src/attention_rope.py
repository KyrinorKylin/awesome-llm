'''
在 attention.py 的基础上加上RoPE编码
'''
import copy
import math
import torch
from torch import nn
from positionalencoding import RotaryPositionalEmbedding


# todo 添加 KV Cache 优化
class MHA(nn.Module):
    '''
    差异：
    1.__init__初始化self.rope
    2.默认用连续位置索引
    3.对 QK 用 RoPE
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k)     # 初始化位置编码

        self.attn_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        :param q: [batch_size, head, seq_len, d_k]
        :param k: [batch_size, head, seq_len, d_k]
        :param v: [batch_size, head, seq_len, d_k]
        :param mask:
        :return: [batch_size, head, seq_len, d_k], [batch_size, head, seq_len, seq_len]
        '''
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        if dropout is not None:
            attn_weights = dropout(attn_weights)

        attn = torch.matmul(attn_weights, v)

        return attn, attn_weights

    def forward(self, q, k, v, mask, token_ids=None):
        '''
        :param q: [batch_size, seq_len, d_model]
        :param k: [batch_size, seq_len, d_model]
        :param v: [batch_size, seq_len, d_model]
        :param mask:
        :param token_ids: [..., seq_len]储存每个token的位置索引
        :return: [batch_size, seq_len, d_model]
        '''
        batch_size, seq_len = q.shape[0], q.shape[1]

        q, k, v = [linear(x) for linear, x in zip(self.linears, (q, k, v))]
        q, k, v = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) for x in [q, k, v]]

        # 如果没有就默认用连续位置索引
        if token_ids is None:
            token_ids = torch.arange(seq_len)

        # 在计算注意力之前对 QK 用 RoPE
        q, k = self.rope(q, token_ids), self.rope(k, token_ids)

        attn, self.attn_weights = self.scaled_dot_product_attention(q, k, v, mask, dropout=self.dropout)

        attn_cat = attn.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.linears[-1](attn_cat)


# todo 添加 KV Cache 优化
class MQA(nn.Module):
    '''
    改变：
    1.初始化
    2.默认用连续位置索引
    3.对 QK 用 RoPE
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)  # 和MHA有一点点不一样，因为MQA的KV只有一个头，所以这里直接映射了。如果还和W_q一样后面用view，那就失去了MQA的意义了。
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.d_k)

        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        q: [batch_size, head, seq_len, d_k]
        k,v: [batch_size, 1, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：[batch_size, head, seq_len, d_k]，[batch_size, head, seq_len, seq_len]
        '''
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        if dropout is not None:
            atten_weights = dropout(atten_weights)

        attentions = torch.matmul(atten_weights, v)

        return attentions, atten_weights

    def forward(self, q, k, v, mask, token_ids=None):
        '''
        q,k,v: [batch_size, seq_len, d_model]
        return: 添加注意力后的所有句子向量：[batch_size, seq_len, d_model]
        '''
        batch_size, seq_len = q.shape[0], q.shape[1]

        q, k, v = [linear(x) for linear, x in zip([self.W_q, self.W_k, self.W_v], (q, k, v))]

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 如果没有就默认用连续位置索引
        if token_ids is None:
            token_ids = torch.arange(seq_len)

        # 在计算注意力之前对 QK 用 RoPE
        q, k = self.rope(q, token_ids), self.rope(k, token_ids)

        attens, self.atten_weights = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        atten_cat = attens.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(atten_cat)


# todo 添加 KV Cache 优化
class GQA(nn.Module):
    '''
    改变：
    1.初始化
    2.默认用连续位置索引
    3.对 QK 用 RoPE
    4.QKV 五维分组而不是 repeat_interleave
    5.mask 维度调整
    6.结果融合
    '''
    def __init__(self, d_model, num_heads, num_groups, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_k = d_model // num_heads
        self.heads_per_group = num_heads // num_groups

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.d_k)

        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q, k, v, mask, dropout: nn.Dropout):
        '''
        q: [batch_size, num_groups, heads_per_group, seq_len, d_k]
        k,v: [batch_size, num_groups, 1, seq_len, d_k]
        return: 注意力矩阵 * v，注意力矩阵：
            [batch_size, num_groups, heads_per_group, seq_len, d_k]，
            [batch_size, num_groups, heads_per_group, seq_len, seq_len]
        '''
        # [batch_size, num_groups, heads_per_group, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        atten_weights = torch.softmax(scores, dim=-1)

        if dropout is not None:
            atten_weights = dropout(atten_weights)

        # [batch_size, num_groups, heads_per_group, seq_len, d_k]
        attentions = torch.matmul(atten_weights, v)

        return attentions, atten_weights

    def forward(self, q, k, v, mask, token_ids=None):
        '''
        q,k,v: [batch_size, seq_len, d_model]
        return: 添加注意力后的所有句子向量：[batch_size, seq_len, d_model]
        '''
        batch_size, seq_len = q.shape[0], q.shape[1]

        q, k, v = [linear(x) for linear, x in zip([self.W_q, self.W_k, self.W_v], (q, k, v))]

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_groups, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_groups, self.d_k).transpose(1, 2)

        # 如果没有就默认用连续位置索引
        if token_ids is None:
            token_ids = torch.arange(seq_len)

        # 在计算注意力之前对 QK 用 RoPE
        q, k = self.rope(q, token_ids), self.rope(k, token_ids)

        # Q 五维分组: [batch_size, num_heads, seq_len, d_k] -> [batch_size, num_groups, heads_per_group, seq_len, d_k]
        q = q.view(batch_size, self.num_groups, self.heads_per_group, seq_len, self.d_k)

        # KV 增加一个维度: [batch_size, num_groups, seq_len, d_k] -> [batch_size, num_groups, 1, seq_len, d_k]
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # 调整 mask 维度：[batch_size, 1, 1, seq_len, seq_len]
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn, self.attn_weights = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 结果融合: [batch_size, num_groups, heads_per_group, seq_len, d_k] -> [batch_size, seq_len, num_groups, heads_per_group, d_k] -> [batch_size, seq_len, d_model]
        attn_cat = attn.permute(0,3,1,2,4).contiguous()
        attn_cat = attn_cat.view(batch_size, seq_len, self.num_heads * self.d_k)

        return self.W_o(attn_cat)


# todo 添加 KV Cache 优化
class MLA(nn.Module):
    '''
    改变：
    1.计算注意力时，矩阵吸收
    '''
    def __init__(self, d_model: int, num_heads: int, d_c_kv: int, d_c_q: int, d_rope: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # 每个头里不带 RoPE 的维度，v 的维度也用它
        self.d_c_kv = d_c_kv                # KV latent 维度，一般远小于 d_model
        self.d_c_q = d_c_q                  # Q latent 维度
        self.d_rope = d_rope                # 解耦 RoPE 的维度

        # Q
        self.W_DQ = nn.Linear(d_model, d_c_q)
        self.W_UQ = nn.Linear(d_c_q, d_model)
        self.W_QR = nn.Linear(d_c_q, num_heads * d_rope)

        # KV
        self.W_DKV = nn.Linear(d_model, d_c_kv)
        self.W_UK = nn.Linear(d_c_kv, d_model)
        self.W_UV = nn.Linear(d_c_kv, d_model)
        self.W_KR = nn.Linear(d_model, d_rope)

        self.rope = RotaryPositionalEmbedding(d_rope)
        self.dropout = nn.Dropout(dropout)
        self.atten_weights = None

    def scaled_dot_product_attention(self, q_c, q_r, c_kv, k_r, mask, dropout: nn.Dropout):
        '''
        q_c:  [batch_size, num_heads, seq_len, d_k]
        q_r:  [batch_size, num_heads, seq_len, d_rope]
        c_kv: [batch_size, seq_len, d_c_kv]
        k_r:  [batch_size, 1, seq_len, d_rope]

        return:
            attentions: [batch_size, num_heads, seq_len, d_k]
            atten_weights: [batch_size, num_heads, seq_len, seq_len]

        矩阵吸收：
            不显式恢复完整的 K、V。
            1. K 路径：
               scores = q_c @ (c_kv @ W_UK.T).T
                      = (q_c @ W_UK) @ c_kv.T

            2. V 路径：
               attn = A @ (c_kv @ W_UV.T)
                    = (A @ c_kv) @ W_UV.T
        '''
        # 将W_UK吸收到 Q 中: [d_model,d_c_kv] -> [num_heads, d_k, d_c_kv]
        W_UK = self.W_UK.weight.view(self.num_heads, self.d_k, self.d_c_kv)

        # [batch_size, num_heads, seq_len, d_k] @ [num_heads, d_k, d_c_kv] = [batch_size, num_heads, seq_len, d_c_kv]
        scores_ = torch.matmul(q_c, W_UK)

        # 内容注意力: [batch_size, num_heads, seq_len, seq_len]
        scores_c = torch.matmul(scores_, c_kv.transpose(-2,-1).unsqueeze(1))

        # RoPE注意力：[batch_size, num_heads, seq_len, seq_len]
        scores_r = torch.matmul(q_r, k_r.transpose(-2,-1))

        # 合并注意力分数
        scores = (scores_c + scores_r) / math.sqrt(self.d_k + self.d_rope)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)

        if dropout is not None:
            attn_weights = dropout(attn_weights)

        # V的路径: [batch_size, num_heads, seq_len, seq_len] @ [batch_size, 1, seq_len, d_c_kv] = [batch_size, num_heads, seq_len, d_c_kv]
        attn_ = torch.matmul(attn_weights, c_kv.unsqueeze(1))

        # W_UV吸收到注意力加权之后: [d_model,d_c_kv] -> [1, num_heads, d_c_kv, d_k]
        W_UV = self.W_UV.weight.view(self.num_heads, self.d_k, self.d_c_kv).transpose(-2,-1).unsqueeze(0)

        # 注意力加权: [batch_size, num_heads, seq_len, d_k]
        attn = torch.matmul(attn_, W_UV)

        # W_UV 有bias，需要单独处理。原始计算中: A @ (c_kv @ W_UV.T + bias)
        if self.W_UV.bias is not None:
            bias = self.W_UV.bias.view(self.num_heads, self.d_k).unsqueeze(0).unsqueeze(2)

            weight_sum = attn_weights.sum(dim=-1, keepdim=True)
            attn += weight_sum * bias

        return attn, attn_weights

    def forward(self, x, mask, token_ids=None):
        '''
        x: [batch_size, seq_len, d_model]
        token_ids: [seq_len]，每个 token 的位置下标
                   不传就默认 0 ~ seq_len-1

        return:
            [batch_size, seq_len, d_model]
        '''
        batch_size, seq_len = x.shape[0], x.shape[1]
        if token_ids is None:
            token_ids = torch.arange(seq_len, device=x.device)

        # Q的低秩分解投影
        c_q = self.W_DQ(x)      #[batch_size, seq_len, d_c_q]
        q_c = self.W_UQ(c_q)    #[batch_size, seq_len, d_model]
        q_r = self.W_QR(c_q)    #[batch_size, seq_len, heads * d_rope]

        # q 的分头
        q_c = q_c.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)        # [batch_size, heads, seq_len, d_k]
        q_r = q_r.view(batch_size, seq_len, self.num_heads, self.d_rope).transpose(1,2)     # [batch_size, heads, seq_len, d_rope]

        # KV 压缩（不投影，不显式还原）
        c_kv = self.W_DKV(x)    #[batch_size, seq_len, d_c_q]

        # 解耦的 k_r
        k_r = self.W_KR(x)      #[batch_size, seq_len, d_rope]
        k_r = k_r.unsqueeze(1)  #[batch_size, 1, seq_len, d_rope]

        # RoPE
        q_r = self.rope(q_r, token_ids)
        k_r = self.rope(k_r, token_ids)

        attn, self.atten_weights = self.scaled_dot_product_attention(q_c, q_r, c_kv, k_r, mask, self.dropout)

        # 结果融合
        attn_cat = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)

        return self.W_O(attn_cat)