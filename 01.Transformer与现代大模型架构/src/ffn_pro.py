'''
只对 DeepSeekMoE 改进
'''
# todo Router能不能像注意力机制那样计算概率
# todo Load Balancing Loss：负载均衡损失
# todo Auxiliary-Loss-Free Load Balancing：无辅助损失负载均衡
# todo Expert Capacity：专家容量限制
# todo Efficient Dispatch：高效分发 Token
import torch
from torch import nn
from ffn import GLU

class DeepSeekRouter(nn.Module):
    '''
    不同：
    1.多返回了一个 logits
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
            logits:   [batch_size, seq_len, num_experts]
        '''
        # 每个 token 对应的专家概率：[batch_size, seq_len, num_experts]
        logits = self.router(x)     
        logits = torch.sigmoid(logits)  #sigmoid 没有dim参数

        # top-k: [batch_size, seq_len, top_k]
        topk_logits, topk_ids = torch.topk(logits, k=self.top_k, dim=-1)

        # 计算归一化权重: [batch_size, seq_len, top_k]
        topk_weights = topk_logits / topk_logits.sum(dim=-1, keepdim=True)

        # 路由权重缩放
        topk_weights = topk_weights * self.route_scale

        return topk_weights, topk_ids, logits


class DSMoEBalancingLoss(nn.Module):
    '''
    不同：
    1.设置 aux_loss_alpha
    2.添加 load_balancing_loss()
    3.forward 计算并返回 loss，1行代码搞定
    '''
    def __init__(self, d_model, d_ff, num_experts, num_shared_experts, top_k, route_scale=2.5, aux_loss_alpha=0.01) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts= num_shared_experts
        self.aux_loss_alpha = aux_loss_alpha

        self.router = DeepSeekRouter(d_model, num_experts, top_k, route_scale)
        self.experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_shared_experts)])

    def load_balancing_loss(self, logits, ids):
        '''
        logits: [batch_size, seq_len, num_experts]
        ids:    [batch_size, seq_len, top_k]

        return:
            aux_loss: 标量
        '''
        batch_size, seq_len, num_experts = logits.shape

        # ----------- 1.计算实际负载占比 fi ----------------
        # 初始化负载统计，每个专家有多少token：[batch_size, num_experts]
        expert_counts = torch.zeros(batch_size, num_experts, device=ids.device, dtype=torch.long)

        # [batch_size, seq_len * top_k]
        ids_flat = ids.view(batch_size, -1)

        # 统计每个专家收到的 token 数量：[batch_size, num_experts]
        expert_counts = torch.scatter_add(dim=1, index=ids_flat, src=torch.ones_like(ids_flat)).float() #按照index的索引，把src对应的值加到expert_counts中

        # fi =  [expert_counts / (top_k * T)] * num_experts
        # [batch_size, num_experts]
        f = (expert_counts / (self.top_k * seq_len)) * num_experts

        # --------- 2.计算平均路由分数 P_i ---------
        # 每个 token 对所有 Expert 的分数进行归一化，保持量级一致：[batch_size, seq_len, num_experts]
        logits = logits.float()
        logits = logits / logits.sum(dim=-1, keepdim=True)

        # 对前2维求平均值：[batch_size, seq_len, num_experts] -> [batch_size, num_experts]
        p = logits.mean(dim=1)

        # --------- 3.计算负载均衡损失 ---------
        aux_loss = self.aux_loss_alpha * torch.sum(f * p, dim=-1)   #[batch_size]
        aux_loss = aux_loss.mean()  # 对 batch_size 求平均，得到标量

        return aux_loss

    def forward(self, x, return_aux_loss=False):
        '''
        x: [batch_size, seq_len, d_model]

        return:
            output: [batch_size, seq_len, d_model]
            aux_loss: 标量（return_aux_loss=True 时额外返回）
        '''
        batch_size, seq_len, d_model = x.shape

        # 展开token:[batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # --------- 1.Routed Experts ---------
        # Routing路由，并展开: [batch_size * seq_len, top_k]
        weights, ids, logits = self.router(x)
        weights_flat = weights.view(-1, self.top_k)
        ids_flat = ids.view(-1, self.top_k)

        # 计算负载均衡损失
        aux_loss = self.load_balancing_loss(logits, ids)

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

        if return_aux_loss:
            return output, aux_loss
        return output


class DeepSeekRouterAuxFree(nn.Module):
    '''
    不同：
    1.__init__ 增加 expert_bias，用于调整 Expert 的路由选择
    2. Top-K 使用 logits + expert_bias
    3. 路由权重仍然使用原始 logits
    '''
    def __init__(self, d_model, num_experts, top_k, route_scale=2.5):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale

        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert Bias：不参与梯度更新
        self.register_buffer('expert_bias', torch.zeros(num_experts))

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        return:
            topk 权重：[batch_size, seq_len, top_k]
            topk 索引：[batch_size, seq_len, top_k]
            logits:   [batch_size, seq_len, num_experts]
        '''
        # 原始路由分数：[batch_size, seq_len, num_experts]
        logits = self.router(x)
        logits = torch.sigmoid(logits)

        # 调整偏置
        adjusted_logits = logits + self.expert_bias

        # top-k: [batch_size, seq_len, top_k]
        _, topk_ids = torch.topk(adjusted_logits, k=self.top_k, dim=-1)

        # 计算权重: [batch_size, seq_len, top_k]
        topk_logits = torch.gather(logits, dim=-1, index=topk_ids)  #按照 index 在指定维度上取值
        topk_weights = topk_logits / topk_logits.sum(dim=-1, keepdim=True)

        # 路由权重缩放
        topk_weights = topk_weights * self.route_scale

        return topk_weights, topk_ids, logits



class DSMoEAuxFree(nn.Module):
    '''
    不同：
    1. 删除 aux_loss_alpha，换成 bias_update_rate=0.001
    2. 删除 load_balancing_loss()，换成 update_expert_bias()
    3. forward 如果是训练状态就更新 bias，2行代码
    '''
    def __init__(self, d_model, d_ff, num_experts, num_shared_experts, top_k, route_scale=2.5, bias_update_rate=0.001) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts= num_shared_experts
        self.bias_update_rate = bias_update_rate

        self.router = DeepSeekRouter(d_model, num_experts, top_k, route_scale)
        self.experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([GLU(d_model, d_ff, activation='silu') for _ in range(num_shared_experts)])

    def update_expert_bias(self, ids):
        '''
        ids: [batch_size, seq_len, top_k]

        根据实际负载动态更新 Expert Bias
        '''
        # 统计每个 Expert的Token数量: [num_experts]
        ids_flat = ids.view(-1)
        expert_count = torch.bincount(ids_flat, minlength=self.num_experts).float()

        # 所有Expert 平均接收的Token数量
        mean_load = expert_count.mean()

        # 更新 expert_bias
        with torch.no_grad():
            self.router.expert_bias.sub_(self.bias_update_rate * torch.sign(expert_count - mean_load))

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]

        return:
            output: [batch_size, seq_len, d_model]
        '''
        batch_size, seq_len, d_model = x.shape

        # 展开token:[batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # --------- 1.Routed Experts ---------
        # Routing路由，并展开: [batch_size * seq_len, top_k]
        weights, ids, logits = self.router(x)
        weights_flat = weights.view(-1, self.top_k)
        ids_flat = ids.view(-1, self.top_k)

        # 利用本次路由结果更新 bias
        if self.training:   #不是人为设置的变量，自己就有
            self.update_expert_bias(ids)

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