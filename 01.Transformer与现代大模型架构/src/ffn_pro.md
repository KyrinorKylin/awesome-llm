## 一、Load Balancing Loss

$$
L_{\text{aux}} = \alpha \sum_{i=1}^{N} f_i P_i
$$

其中：

- (N)：Routed Expert 的数量。
- (K)：每个 Token 激活的专家数量（Top-K）。
- (T)：当前参与路由的 Token 总数。
- (f\_i)：专家 (i) 实际接收到的 Token 占比。
- (P\_i)：Router 分配给专家 (i) 的平均路由分数。
- (\alpha)：辅助损失的系数，它的作用是：控制模型有多大程度愿意为了负载均衡而调整 Router

### 1. 实际负载占比 (f\_i)

$$
f_i = \frac{n_i}{T \times K} \times N
$$

其中：

- (n\_i)：专家 i 接收到的 Token 数量。
- (T)：Token 总数。
- (K)：每个 Token 选择的专家数量。
- (N)：专家总数。

T\*K 表示总共分配了多少次专家，$\frac{n_i}{T \times K}$ 表示该专家的负载占比

为什么要乘以N呢？

因为如果负载均衡，那么一个专家的选择次数应该为 $\frac{TK}{N}$，所以负载占比 = $\frac{TK}{N*TK}$ = $\frac{1}{N}$

要比较 $\frac{n_i}{T \times K}$ 和 $\frac{1}{N}$ 的关系，就相当于比较 $\frac{n_i}{T \times K} \times N$ 和 1 的关系，乘以N是为了归一化方便比较

### 2. 平均路由分数 (P\_i)

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{t,i}
$$

其中 (s\_{t,i}) 是 Token (t) 对专家 (i) 的 Sigmoid 路由分数。

这里使用的是 Top-K 选择之前的全部 Sigmoid 分数，而不是 Top-K 归一化后的权重。

这样可以让未被选中的专家也通过路由分数获得梯度。而且 f\_i 是不可导的（topk 是离散操作），但 P\_i 可导，梯度通过 P\_i 回传给 router

(P\_i)还有一个好处：他和 fi 相乘，如果 fi 本身很低，Pi 也会很低，梯度更新就不会继续压低它的路由分数，如果 fi 很高也是一样的。

因此，通过最小化这个损失，可以推动 Router 减少对高负载专家的分数，并增加对低负载专家的分数

***

## 二、Auxiliary-Loss-Free Load Balancing 无辅助损失负载均衡

它的核心思想是：
不再通过额外的 Auxiliary Loss（辅助损失）迫使 Router 学习均衡分配，而是动态调整每个 Expert 的路由偏置（Expert Bias），直接影响 Router 的 Top-K 选择。

### 传统 Load Balancing Loss 有什么问题？

1. 传统做法是增加负载均衡损失，但问题在于辅助损失会干扰模型原本的优化目标。例如：

- 模型认为某些 Token 特别适合 Expert 1。
- 为了负载均衡，Auxiliary Loss 又要求 Router 减少分配给 Expert 1 的 Token。
- 两个目标可能发生冲突，干扰模型本身的优化方向

2. 随着模型规模增大，过强的均衡约束反而会损害专家的专业化程度

### Auxiliary-Loss-Free 的核心原理

在 Router 的线性输出上额外加了一个长度为 num\_experts 的可训练量 bias，但它不参与梯度更新（requires\_grad=False），只在每个 step 结束后按下面的规则手动刷新：

- Top-K 选择阶段：logits = logits + bias，用加完偏置后的分数做 topk，从而影响"哪些专家被选中"；
- 权重计算阶段：依然使用原始 sigmoid 分数（不加 bias），做归一化后再乘 route\_scale，保证门控值不被偏置污染；
- 偏置更新：每步根据当前 batch 中每个专家实际接收的 token 数 c\_i 与平均负载 c̄ 的差值符号来调整：b\_i ← b\_i + γ · sign(c̄ - c\_i)

其中 γ（bias\_update\_speed）是固定的更新步长，超载专家会被减、欠载专家会被加，最终让所有专家的负载趋向均衡

### 和传统 Load Balancing Loss 有什么区别？

| 对比 | Load Balancing Loss | Auxiliary-Loss-Free |
| --- | --- | --- |
| 核心机制 | 增加辅助损失 | 动态调整 Bias |
| 是否修改训练 Loss | 是 | 否 |
| 是否影响梯度 | 是 | 不直接影响梯度 |
| 是否修改 Top-K 选择 | 间接影响 | 直接影响 |
| 是否需要额外参数 | 不需要 Bias | 需要 Expert Bias |
| 主要目标 | 均衡负载 | 均衡负载 |

