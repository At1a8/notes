# Parallelizing Linear Transformers with the Delta Rule over Sequence Length

相关博客: 

https://sustcsonglin.github.io/blog/2024/deltanet-1/

https://kexue.fm/archives/11033

论文: https://arxiv.org/pdf/2406.06484

视频: https://www.bilibili.com/video/BV1Pg5JzEEad/

## Linear attention as RNN
符号说明：我们使用大写粗体字母表示矩阵，小写粗体字母表示向量，普通小写字母表示标量。

### What is linear attention?
虽然传统的softmax注意力机制功能强大，但它的复杂度与序列长度呈二次方关系。让我们从标准的softmax注意力机制（假设只有一个注意力头）入手，看看线性注意力机制是如何解决这个问题的：

$`
\begin{aligned} 
\mathrm{Parallel\ training:} &&& \mathbf{O} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d} \\ 
\mathrm{Iterative\ inference:} &&&\mathbf{o_t} = \sum_{j=1}^t \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^t\exp(\mathbf{q}^\top_t \mathbf{k}_l)}\mathbf{v}_j &&\in \mathbb{R}^d 
\end{aligned}
`$

这里，

$L$表示序列长度

$d$代表注意力头的维度

$\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O} \in \mathbb{R}^{L \times d}$分别表示query矩阵、key矩阵、value矩阵和output矩阵。

$\mathbf{M} \in \mathbb{R}^{L \times L}$ 它是自回归建模的因果掩码，确保每个位置只能关注先前的位置。

$\boldsymbol{q}_i,\boldsymbol{k}_i,\boldsymbol{v}_i,\boldsymbol{o}_i \in \mathbb{R}^{d\times 1}$

线性注意力做的只是简单地移除了 softmax 运算符：

$`
\begin{aligned} \mathrm{Parallel\ training：} &&&\mathbf{O}= (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d} \\ 
\mathrm{Iterative\ inference：}&&&\mathbf{o}_t = \sum_{j=1}^t (\underbrace{\mathbf{q}_t^\top}_{[1,d_k]} \underbrace{\mathbf{k}_j}_{[d_k,1]}) \underbrace{\mathbf{v}_j}_{[d_v,1]} &&\in \mathbb{R}^d \end{aligned}
`$

虽然移除 softmax 函数本身并不能立即降低计算复杂度，但它却能实现一个至关重要的数学性质：**线性化**。这和结合律使我们能够以显著提高效率的方式重构计算过程。对于训练过程，研究人员已经开发出了分块并行技术。它利用这种线性特性，在保持硬件效率的同时实现亚二次复杂度。

对于推理，我们还可以按如下方式重新排列计算（$` \mathbf{q}_t^\top\mathbf{k}_j `$ 等价于 $` \mathbf{k}_j^\top \mathbf{q}_t `$ 都是 1 * 1 的标量）：

$`
\begin{aligned} &&&&\mathbf{o_t} = \sum_{j=1}^t \mathbf{v}_j(\mathbf{k}_j^\top \mathbf{q}_t) &&&&& \mathbf{k}_j^\top \mathbf{q}_t = \mathbf{q}_t^\top \mathbf{k}_j \in \mathbb{R}\\ &&&&= (\sum_{j=1}^t\mathbf{v}_j\mathbf{k}_j^\top)\mathbf{q}_t &&&&&\text{By associativity} \end{aligned}
`$

```
注:推理和训练的主要区别在于:
训练时所有QKV均为已知，QKV的矩阵形式涵盖了一个序列的所有token，因此还有M矩阵来确保矩阵的下三角化（因果掩码）

Q[L,d] × K[L,d]ᵀ  →  Score[L,L]  →  Out[L,d]
↑一次性看到所有token

推理时只能确定当前序列已经推理出的部分，后续token为未知量，单个序列会以向量/张量的形式存在

q[1,d] × K_cache[t,d]ᵀ → Score[1,t] → Out[1,d]
↑只生成当前token，复用cache
```

定义一个状态矩阵：$` \mathbf{S}_t = \sum_{j=1}^t\mathbf{v}_j\mathbf{k}_j^\top `$ 那么，它的计算可以表示为：

$`
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top \in \mathbb{R}^{d\times d}, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t \in \mathbb{R}^{d}
`$

这种表述表明，线性注意力机制本质上是一个具有矩阵值状态 $\mathbf{S}$ 的线性循环神经网络。 
它累积key-value外积，从而能够实现高效地状态扩展(从$`\mathcal{O}(d)$到$\mathcal{O}(d^2)`$)。

采用这种方法，我们只需要存储和更新 $\mathbf{S}_t$ 而不是保留所有先前的键值对。这种优化显著提高了效率：自回归推理的时间复杂度从$`\mathcal{O}(L^2d)`$降低至$`\mathcal{O}(Ld^2)`$，同时，空间复杂度从$`\mathcal{O}(Ld)`$降低至$`\mathcal{O}(d^2)`$。这些改进使得该方法在以下两种情况下尤为有利：

- 长序列建模中，softmax 注意力机制的二次复杂度可能是一个重要的瓶颈。

- 在生成过程中，decode阶段常常是memory-bound，在$`L>>d`$的场景移除kvcache可以显著降低推理延迟。

### Key Limitations of Linear Attention
线性注意力机制中固定大小的状态矩阵意味着它无法完美地保留所有历史信息，这使得精确检索变得尤为困难。

更正式地说，线性注意力机制实现了一种键值关联记忆，它是键和值之间外积的总和。$`\mathbf{S} = \sum \mathbf{v}_i\mathbf{k}_i^\top`$。假设所有key都已规范化为单位长度，当我们尝试检索与特定key $`k_j`$关联的value时，我们得到：

$`
\begin{aligned} \mathbf{S}\mathbf{k}_j &= \sum \mathbf{v}_i (\mathbf{k}_i^\top \mathbf{k}_j) \\ &= \mathbf{v}_j + \underbrace{\sum_{i\neq j} (\mathbf{k}_i^\top \mathbf{k}_j)\mathbf{v}_i}_{\text{retrieval error}} \end{aligned}
`$

为了最大限度地减少检索误差项，我们需要对所有$`i\neq j`$都能满足$`\mathbf{k}_i^\top \mathbf{k}_j = 0`$。换句话说，所有键都应该相互正交。然而，这揭示了一个根本性的局限性：在$`d`$-维空间中，你最多只能有$`d`$个正交向量。这就解释了为什么增加head dim会有帮助, 它在向量空间中提供了更多“空间”，用于存储不同的key-value对。

这种理论局限性在实践中得到了体现：在语言建模中，传统的线性注意力机制与softmax注意力机制相比表现不佳（差距很大）。主要原因是内存“过载”：在这种键值关联记忆系统中，我们只能添加新的键值关联，而无法删除已有的信息。随着序列长度的增加，这会导致“检索错误”的累积，从而降低性能。

线性注意力机制的门控变体（例如 GLA和mamba）的最新进展中通过引入**遗忘机制**，显著缩小了与标准注意力机制在语言建模任务中的性能差距。然而，这些模型在上下文检索和精确复制能力方面仍然面临着根本性的挑战——这些局限性在近期的研究中已得到实证观察和理论证明。

#### 关于 retrieval error 的数值举例
设QKV维度为2维
```
k1 = [1, 0]^T
k2 = [1, 1]^T
v1 = [10, 0]^T
v2 = [0, 5]^T
```
**step1 构造 memory**

$`
\mathbf{S} = \mathbf{v}_1\mathbf{k}_1^\top + \mathbf{v}_2\mathbf{k}_2^\top
`$

```
先算外积：
v1 * k1^T 
= [10, 0]^T × [1, 0]
= [[10, 0],
   [0, 0]]

v2 * k2^T 
= [0, 5]^T × [1, 1]
= [[0, 0],
   [5, 5]]

加起来：
S = [[10, 0],
     [5, 5]]
```

**step2 用 k1 读取**
```
S * k1 
= [[10, 0], [5, 5]] × [1, 0]^T
= [10, 5]^T

但真实 value 应该是：v1 = [10, 0]^T
结果变成：[10, 5]^T，这是 v2 的干扰。
```
**为什么会干扰**

因为：k1 · k2 = 1 而不是 0，所以 $`(\mathbf{k}_2^\top\mathbf{k}_1)\mathbf{v}_2`$ 会被加进来。
如果 keys 正交：k1 · k2 = 0，那么就没有 retrieval error。
```
S * k1 = v1
S * k2 = v2
```

## DeltaNet: Linear Attention with Delta Rule
### What is Delta Rule?
Delta Rule是神经网络中一个基本的误差纠正学习原则。它的核心思想非常简单：根据我们想要的结果（target）和实际得到的结果（pred）之间的差异（delta）来调整模型的参数。

为了更直观地理解这一点，想象一下教孩子瞄准目标。如果他们射偏到了左边，你就告诉他们向右调整；如果向右偏了，就向左调整。调整的幅度取决于他们偏离目标的距离——这个概念直接体现在“Delta Rule”中。

### What is DeltaNet?
DeltaNet将这种纠错原理应用于线性注意力机制。它不再简单地累积键值外积，而是基于预测误差来更新自身状态：

$`
\begin{align} \mathbf{S}_{t} &= \mathbf{S}_{t-1} - \beta_t(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top 
\\
&= \mathbf{S}_{t-1} - \beta_t \mathbf{S}_{t-1} \mathbf{k}_t \mathbf{k}_t^\top + \beta_t \mathbf{v}_t \mathbf{k}_t^\top \end{align}
`$

当我们把各个组成部分拆解开来时，它与Delta Rule的相似之处就显而易见了：

$`\beta_t \in \mathbb{R}`$充当学习率

$`\mathbf{k}_t \in \mathbb{R}^d`$是输入数据

$`\mathbf{v}_t \in \mathbb{R}^d`$是目标

$`\mathbf{S}_{t-1} \mathbf{k}_t \in \mathbb{R}^d`$这是我们目前的预测

还有另一种直观的方法来理解这条更新规则。考虑$`\mathbf{S}_{t-1}\mathbf{k}_t`$为从内存中检索到的 **和当前key $`\mathbf{k}_t`$ 关联的旧value**。对于同一个key，当我们遇到一个新关联的value $`\mathbf{v}_t`$ 时，我们不会盲目覆盖，而是进行谨慎的更新：

$`
\begin{align} \mathbf{v}_t^{\text{new}} &= (1-\beta_t) \mathbf{v}_t^{\text{old}} + \beta_t \mathbf{v}_t, \\ \mathbf{S}_t &= \mathbf{S}_{t-1} - \underbrace{\mathbf{v}_t^{\text{old}} \mathbf{k}_t^\top}_{\text{erase}} + \underbrace{\mathbf{v}_t^{\text{new}} \mathbf{k}_t^\top}_{\text{write}} \end{align}
`$

在这里 $`\mathbf{v}_t^{\text{new}}`$ 是旧value和新value的组合，由动态因素 $`\beta_t \in (0,1)`$ 控制：
- 当 $`\beta_t=0`$, 内存内容保持不变
- 当 $`\beta_t=1`$, 我们将关联的旧value完全替换为新的value

### Why is DeltaNet Superior at In-context Retrieval Compared to Linear Attention?
DeltaNet的更新规则可以通过在每个时间步 $`t`$ 使用梯度下降法依次最小化期望输出和预测输出之间的均方误差（MSE）来推导得出。

$`
\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S} \mathbf{k}_t - \mathbf{v}_t\|^2
`$

应用梯度下降法最小化均方误差损失得到：

$`
\begin{aligned} \mathbf{S}_t &= \mathbf{S}_{t-1} - \eta_t \nabla \mathcal{L}_t(\mathbf{S}_{t-1}) \\ &= \mathbf{S}_{t-1} - \eta_t \left(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t\right) \mathbf{k}_t^\top \end{aligned}
`$

学习率 $`\eta_t`$ 设置为 $`\beta_t`$，即得到 DeltaNet 的更新规则。
相比之下，传统的线性注意力机制采用的是线性损失函数：

$`
\mathcal{L}^\prime_t(\mathbf{S}) = -\langle \mathbf{S} \mathbf{k}_t, \mathbf{v}_t \rangle
`$

线性注意力机制对应的更新规则为：

$`
\begin{aligned} \mathbf{S}_t &= \mathbf{S}_{t-1} - \eta_t \nabla \mathcal{L}_t^\prime(\mathbf{S}_{t-1}) \\ &= \mathbf{S}_{t-1} + \eta_t \mathbf{v}_t \mathbf{k}_t^\top \end{aligned}
`$

$`\eta_t = 1`$ 即可得到标准的线性注意力更新。

因此，DeltaNet 在上下文检索方面的卓越性能显而易见->它在每一步都最大限度地减少了 MSE，使其成为联想回忆等任务的理想选择，因为减少大错误对于准确检索至关重要。