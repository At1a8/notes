# Parallelizing Linear Transformers with the Delta Rule over Sequence Length

作者博客: https://sustcsonglin.github.io/blog/2024/deltanet-1/

论文: https://arxiv.org/pdf/2406.06484

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