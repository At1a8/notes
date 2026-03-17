import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,        # 服务器配置参数
        gpu_id: int,                    # 当前 GPU 设备 ID
        tp_rank: int,                   # Tensor Parallel rank（张量并行 rank）
        dp_rank: Optional[int],         # Data Parallel rank（数据并行 rank，可选）
        moe_ep_rank: int,               # MoE Expert Parallel rank（专家并行 rank）
        attn_cp_rank: int,              # Attention Context Parallel rank（注意力上下文并行 rank）
        moe_dp_rank: int,               # MoE Data Parallel rank（MoE 数据并行 rank）
        nccl_port: int,                 # NCCL 通信端口
        target_worker: TpModelWorker,   # 目标模型（大模型）的 worker
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk                                      # 每步候选分支数
        self.speculative_num_steps = server_args.speculative_num_steps                      # 推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens        # 总草稿 token 数
        self.enable_nan_detection = server_args.enable_nan_detection                        # 是否检测 NaN
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size                                              # KV cache page size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm                                               # 算法类型：EAGLE / EAGLE2 / EAGLE3 / MTP 等
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len    # Draft 模型需要能和 Target 模型处理同样长的序列，确保推测解码能覆盖完整上下文

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph                          # 延迟生成 cuda graph，主要原因是父类 TpModelWorker 会默认生成 cuda graph
        server_args.disable_cuda_graph = True                                               # 但 EAGLE 需要先完成 draft model 初始化，再手动完成 EAGLE CUDA Graph
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (                         # draft model 共享 target model 的 kv cache pools, 但各自独立存储
            target_worker.get_memory_pool()                                                 # req_to_token_pool: 请求到 token 的映射池, token_to_kv_pool_allocator: KV cache 分配器
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():                                          # EAGLE3 模型自带 hot token map，忽略外部配置
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None                                                        # 延迟到后面从模型获取
        elif server_args.speculative_token_map is not None:                                 # EAGLE/EAGLE2：从外部文件加载热词表
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'                           # 覆盖模型配置，限制词表大小
            )
        else:
            self.hot_token_id = None                                                        # 不使用热词表，热词表作用是限制 Draft Model 的输出词表到高频词（如 top-1024），加速采样。

        # Init draft worker，初始化 Draft Worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():      # DP Attention + EAGLE3 需要特殊 TP 上下文
            ctx = draft_tp_context(get_attention_tp_group())
        else:                                                                               # 普通上下文
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():        # 调用父类 TpModelWorker 初始化，加载 Draft Model
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,                                                       # is_draft_worker=True 告诉父类这是 Draft Model，会加载对应的模型架构（如 Qwen3_5ForCausalLMMTP）
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()            # 参数共享（Embedding & LM Head），从 Target Model 获取 embedding 和 lm_head

        if self.speculative_algorithm.is_eagle3():                                          # EAGLE3 默认不共享 lm_head，但某些模型例外
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)               # 特例：如 nvidia/gpt-oss-120b-Eagle3 共享 lm_head
            else:
                self.draft_model_runner.model.set_embed(embed)                              # 标准 EAGLE3：只共享 embedding

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(          # 从模型获取热词表（EAGLE3 内嵌）
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]                                    # 如果使用了热词表，裁剪 lm_head 到热词子集，只保留热词对应的权重

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)                   # EAGLE/EAGLE2/MTP：共享 embedding + lm_head
        """
        参数共享策略：
        EAGLE3: Embedding 共享, LM Head 独立，模型自带输出头
        EAGLE/EAGLE2/MTP: Embedding 和 LM Head 都共享，完全共享，热词表做裁剪
        目的：减少内存占用，保持 Draft 和 Target 的语义一致性
        """

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph                                                       # 恢复原始的 CUDA Graph 设置
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context          # 设置 Draft TP 上下文（用于 DP Attention）
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True                                          # 从模型配置读取是否使用辅助隐藏状态
            eagle_config = getattr(                                                         # 辅助隐藏状态：EAGLE3 可能使用额外的 hidden states 作为 Draft Model 输入，增强预测能力。
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()                                                   # 创建多步注意力后端（支持树形注意力）
            self.init_cuda_graphs()                                                         # 捕获 Draft 和 Draft-Extend 的 CUDA Graphs

        # Some dummy tensors                                                                # 在 _draft_preprocess_decode 中用于大页内存分配计算，避免运行时频繁创建张量。
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,                                                               # 服务器配置（包含 attention 相关参数）
            self.draft_model_runner,                                                        # draft 模型的 runner（包含模型信息和 TP group）
            self.topk,                                                                      # 每步候选分支数（如 4）
            self.speculative_num_steps,                                                     # 推测步数（如 5）
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()             # decode阶段: draft model 需要多步树形注意力, 不同于标准解码注意力的单步自回归

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()                             # prefill阶段: forward_draft_extend & forward_draft_extend_after_decode 
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None                                                       # Draft Decode 阶段的 CUDA Graph（多步树形生成）
        self.cuda_graph_runner_for_draft_extend = None                                      # Draft Extend 阶段的 CUDA Graph（初始化/更新状态）

        if self.server_args.disable_cuda_graph:                                             # --disable-cuda-graph 禁用了 CUDA Graph
            return

        Device2DraftCudaGraphRunner = {                                                     # 同时支持 npu & gpu
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:                                                  # 只有当推测步数大于 1 时才捕获 Draft Decode CUDA Graph
            tic = time.perf_counter()                                                       # 只生成 1 个候选 token，CUDA Graph 收益小, 多步生成，需要重复执行，Graph 优化价值大
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)                 # memory 与 latency 监控
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device                                                   # 使用 self.target_worker.device 而非 self.device，确保与目标模型一致
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:                                  # 存在独立的 extend 注意力后端（某些配置可能为 None），非 NPU 设备（NPU 暂不支持 extend 的 CUDA Graph）
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(      # 创建 Extend Graph Runner，Extend 阶段的 Graph 相对简单（单步，无树形结构），目前只有 CUDA 实现，NPU 跳过
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.        # batch.forward_mode.is_extend(): 当前 batch 处于 Extend 模式（新请求或前缀扩展）
        """                                                                                 # batch.is_extend_in_batch: 混合 batch 中包含 extend 请求（即使主模式是 decode）
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:                      # Extend 阶段需要特殊处理：Target Model 先执行，Draft Model 初始化
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(       # step1: Target Model 处理 prompt
                batch                                                                       # step1: 返回 logits_output（含 full hidden states）、next_token_ids（首 token）、seq_lens_cpu
            )
            with self.draft_tp_context(                                                     # draft_tp_context: 处理 DP Attention 的张量并行上下文
                self.draft_model_runner.tp_group                                            # speculative_moe_backend_context: MoE 后端上下文
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():    # speculative_moe_a2a_backend_context: MoE all-to-all 通信上下文
                self.forward_draft_extend(                                                  # step 2: 使用 Target 的 hidden_states 初始化 Draft Model
                    batch,                                                                  # step 2: 让 Draft Model 的 KV cache 与 Target 对齐, 准备 Draft 的初始状态供后续 decode 使用
                    logits_output.hidden_states,                                            # Target 的 hidden states
                    next_token_ids,                                                         # Target 生成的首 token
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,                                          # ← 多模态输入（如有）
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,                                                      # Extend 是初始化阶段，没有推测-验证循环, 无接受 token
                can_run_cuda_graph=False,                                                   # Extend 阶段输入长度变化大，不适合 CUDA Graph
            )
        else:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group                                            # Draft Model 生成 speculative_num_steps 步候选
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():    # 每步生成 topk 个分支，形成树形结构
                spec_info = self.draft(batch)                                               # 返回 spec_info: EagleVerifyInput（包含树形掩码、候选 tokens 等）
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (        # Target Model 并行验证所有候选路径, 比较 Draft 和 Target 的 logits，决定接受哪些 token
                self.verify(batch, spec_info)                                               # logits_output: Target 的输出（用于后续生成 logprob 等）, verify_output: EagleVerifyOutput: 验证结果（接受的 tokens、长度等）
            )                                                                               # model_worker_batch: 模型 worker batch（用于 overlap schedule）, can_run_cuda_graph: 是否使用了 CUDA Graph

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():    # 准备下一轮 Draft
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention                                    # enable_dp_attention: DP 模式下需要同步所有 rank
                    or batch.spec_info.verified_id.shape[0] > 0                             # verified_id.shape[0] > 0: 还有 token 被接受（生成未结束）
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)                           # 基于验证结果更新 Draft Model 的状态 && 为下一轮 draft() 调用准备初始 hidden states 和 top-k 概率 && 恢复 batch 状态（如 seq_lens 等）

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,                                   # 验证后接受的 tokens
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),           # 总接受数
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,          # 每请求接受数
                can_run_cuda_graph=can_run_cuda_graph,                                      # 是否用了 CUDA Graph
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0                       # batch.spec_info.verified_id: Verify 阶段被接受的 token IDs, shape[0] > 0: 本地有 token 被接受，生成未结束
        if not self.server_args.enable_dp_attention:                                        # 如果没有启用 DP Attention，直接返回本地结果，无需同步。非 DP 模式下，每个 rank 独立处理自己的请求，不需要关心其他 rank 的状态
            return local_need_forward                                                       # local_need_forward 是布尔值，表示当前 rank 是否需要继续 forward

        global_need_forward = torch.tensor(                                                 # 将本地布尔值包装成 1 元素的 int64 张量，用于后续的 all_reduce 通信
            [
                (local_need_forward),
            ],
            dtype=torch.int64,                                                              # 用 int64 而非 bool，因为 all_reduce 对整数支持更好
        )
        torch.distributed.all_reduce(                                                       # all_reduce: 将所有 rank 的值求和，广播到所有 rank
            global_need_forward, group=get_tp_group().cpu_group                             # get_tp_group().cpu_group: 使用 TP (Tensor Parallel) 组的 CPU 通信组
        )                                                                                   # 执行完成后 -> global_need_forward[0] = 需要 forward 的 rank 数量
        """
        Rank 0: local_need_forward = True  (1) ──┐
        Rank 1: local_need_forward = False (0) ──┼
        Rank 2: local_need_forward = True  (1) ──┘
        Rank 3: local_need_forward = False (0) ──┘

        结果: global_need_forward = [2]  (2 个 rank 需要继续)
        """
        # DP Attention 的要求：所有 rank 必须同步执行，保持一致的 batch 状态 && 如果 Rank 0 有请求要继续，Rank 1 即使没有请求，也要参与 forward（可能是空操作） && 否则会导致分布式死锁或状态不一致
        global_need_forward_cnt = global_need_forward[0].item()                             # global_need_forward_cnt: 需要 forward 的 rank 数量
        need_forward = global_need_forward_cnt > 0                                          # need_forward = global_need_forward_cnt > 0: 只要有任意一个 rank 需要继续，所有 rank 都要参与

        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.                                                           # 负责目标模型（Target Model）扩展/预填充阶段
                                                                                            # 在推测解码的 Extend 阶段，必须由 Target Model 先执行，为 Draft Model 提供初始化条件
        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.        # 需要 full hidden states 来为 Draft Model 的 KV cache 做预填充（prefill）
        model_worker_batch = batch.get_model_worker_batch()                                 # 将 ScheduleBatch 转换为 ModelWorkerBatch，这是底层模型执行的标准输入格式
        """
        FULL: 获取所有位置的 hidden states, Extend 阶段必需：Draft 需要完整序列信息
        LAST: 获取最后一个位置的 hidden states, Decode 阶段，只需最新状态
        NONE: 不获取 hidden states, 纯推理，无后续处理需求
        输入序列: [t0, t1, t2, t3, t4] (prompt tokens)
          ↓
        Target Model 前向传播
          ↓
        Hidden states: [h0, h1, h2, h3, h4]  ← FULL 模式捕获全部
          ↓
        Draft Model 使用 [h0, h1, h2, h3, h4] 预填充自己的 KV cache
        使得 Draft 能从 t4 位置继续生成 t5, t6, t7...
        """
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)      # self.target_worker: 外部传入的目标模型 worker（大模型）
        logits_output, next_token_ids = (                                                   # forward_batch_generation: 标准生成接口，执行实际的前向传播
            batch_result.logits_output,                                                     # logits_output	包含 hidden_states（FULL 模式下的全部）、logits、可能的 mm_input_embeds -> 给 Draft Model 做初始化
            batch_result.next_token_ids,                                                    # Target Model 生成的第一个 token（t5）-> 作为生成起点
        )
        return (
            logits_output,                                                                  # 包含 full hidden states
            next_token_ids,                                                                 # 首个生成的 token
            model_worker_batch.seq_lens_cpu,                                                # 序列长度（CPU 张量）
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):                               # 为 Draft 阶段的 Decode 做预处理，核心是分配和管理 KV cache，支持多步、多分支（top-k）的推测生成
        batch.maybe_evict_swa()                                                             # 如果启用滑动窗口注意力(SWA)，需要放弃旧的 KV cache 条目
        for req in batch.reqs:
            req.decode_batch_idx += 1                                                       # 记录每个请求经历的 decode 批次次数，用于日志或调度分析

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:                          # 频率惩罚、存在惩罚
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(              # 将上一轮验证接受的 token 加入惩罚统计
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:                                                             # Page Size = 1（简单情况）
            alloc_len_per_decode = self.speculative_num_steps * self.topk                   # 每序列需要 steps * topk 个位置
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc            # 实际上只需要 (steps - 1) * topk，因为第一个 token 已存在，这里简化处理
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(               # alloc_token_slots：从内存池分配连续 token 槽
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:                                                                               # Page Size > 1（复杂情况，PagedAttention）
            if self.topk == 1:                                                              # 子情况 B1：Top-k = 1（单分支），单分支时，只需线性扩展，无需处理分支复制
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:                                                                           # 子情况 B2：Top-k > 1（多分支，最复杂）
                # In this case, the last partial page needs to be duplicated.               # 大页（page_size > 1）+ 多分支（topk > 1）时，最后一个部分填充的页需要复制多份
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    top-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens
                """
                假设 page_size = 4, topk = 3, 当前 seq_len = 10

                物理 KV cache 布局：
                [0,1,2,3] [4,5,6,7] [8,9,_,_]  ← 最后一页只有 2 个有效 token (8,9)

                为 3 个分支各分配新页：
                [0,1,2,3] [4,5,6,7] [8,9,_,_] [10,11,12,13]  ← top-k=0 分支
                                        ↓ 复制 8,9
                                    [8,9,_,_] [14,15,16,17]  ← top-k=1 分支  
                                        ↓ 复制 8,9
                                    [8,9,_,_] [18,19,20,21]  ← top-k=2 分支
                """
                
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size                       # 最后一页已有 token 数
                num_new_pages_per_topk = (                                                  # 每个分支需要的新页数（向上取整）         
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1    # 额外的 self.page_size - 1 是为了免去除完再向上取整
                ) // self.page_size
                seq_lens_cpu = (                                                            # 扩展后的总长度（考虑多分支复制）
                    prefix_lens_cpu // self.page_size * self.page_size                      # prefix 向下取整
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(                                             # 大页场景下的分配函数，处理复杂的页表管理
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:                                            # 处理大页复制（page_size > 1 && Top-k > 1 时）
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)    # 需要复制的 KV 条目总数
            target_cache_loc = torch.zeros(                                                 # 目标位置（新分配的页）
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(                                                 # 源位置（原始最后一页）
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        # 1. 更新 req_to_token 映射表
        # 2. 为每个分支分配独立的页序列
        # 3. 记录需要复制的 KV 位置
        assign_draft_cache_locs[(num_seqs,)](                                               # Triton Kernel 执行 Cache 位置分配, [(num_seqs,)] → 每序列一个 block 并行处理
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:                                            # 执行 KV 复制和清理
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(                     # 物理复制：将原始最后一页的 KV 值复制到各分支的新页
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps                         # 裁剪：移除为对齐页大小而分配的额外 padding，只保留实际需要的 steps * topk 个位置
            ]

        batch.out_cache_loc = out_cache_loc                                                 # 分配的 KV cache 位置，供 Draft forward 使用
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()                               # 总序列长度（用于注意力计算）
        batch.return_hidden_states = False                                                  # 不返回 hidden states
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)            # 每个分支的起始位置（[len0, len0, len0, len1, len1, len1, ...] 重复 topk 次）
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)        # 恢复分配器状态：为可能的回滚或重试做准备（推测解码验证失败时需要）

    # Data Parallel Attention 场景：
    # 多个 GPU (ranks) 并行处理不同请求, 但某些时刻，某个 rank 可能没有分配到请求, 此时该 rank 处于 "idle" 状态
    # 分布式训练/推理需要所有 rank 同步执行, 如果 rank 0 有数据要处理，rank 1 空闲, rank 1 不能真的"什么都不做"，否则会死锁
    def _draft_preprocess_idle(self, batch: ScheduleBatch):                                 # 处理 DP (Data Parallel) Attention 模式下的 idle（空闲）batch
        batch.spec_info = EagleDraftInput.create_idle_input(                                # 生成一个"空"的输入对象
            device=self.device,
            hidden_size=self.model_config.hidden_size,                                      # 从模型配置获取，如 4096、8192 等
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,                                     # LAST，捕获最后位置的 hidden state，即使 idle 也保持与其他模式一致
        )

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():                                                    # DP Attention 下某些 rank 无数据
            self._draft_preprocess_idle(batch)                                              # 创建 dummy 输入，保持同步
        else:                                                                               # 正常生成
            self._draft_preprocess_decode(batch)                                            # 分配 KV cache、更新位置、准备树形结构

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)                                       # 确保预处理后 spec_info 是正确的 EagleDraftInput 类型

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST                              # 只捕获最后位置 hidden state（多步传递用）
        spec_info.num_tokens_per_req = self.topk                                            # 每请求生成 topk 个候选
        spec_info.num_tokens_for_logprob_per_req = self.topk                                # 计算 topk 个 logprob
        batch.return_hidden_states = False                                                  # 不返回 hidden states

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()                                 # ScheduleBatch → model_worker_batch
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST             # 只捕获最后位置 hidden state
        forward_batch = ForwardBatch.init_new(                                              # model_worker_batch -> forward_batch
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(         # 判断是否能用 cuda graph 加速（需满足 batch 大小、序列长度等条件）
            forward_batch
        )
        if can_cuda_graph:                                                                  # cuda graph 可用
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(    # 直接 replay 预录制的图，返回树形结构参数
                forward_batch
            )
        else:                                                                               # cuda graph 不可用
            forward_batch.can_run_dp_cuda_graph = False                                     # 标记不使用 DP CUDA Graph
            if (
                not forward_batch.forward_mode.is_idle()                                    # 多步且非 idle 时，初始化注意力后端的 forward metadata（树形注意力用）
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            parent_list, top_scores_index, draft_tokens = self.draft_forward(               # 调用 draft_forward 执行实际多步前向传播
                forward_batch
            )

        if batch.forward_mode.is_idle():                                                    # idle 提前返回：DP 模式下无数据的 rank 返回 dummy 输入，保持同步
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,                                                                      # 树形因果注意力掩码
            position,                                                                       # 每个节点的位置编码
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(                                                    # 将线性生成的 tokens 转换为树形结构，供 Target Model 并行验证
            spec_info.verified_id,                                                          # 起点 token（来自上一轮验证）
            parent_list,                                                                    # 树形父节点关系
            top_scores_index,                                                               # top-k 选择索引
            draft_tokens,                                                                   # 原始草稿 tokens
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,                                     # 关键：Verify 步骤需要 FULL 模式记录所有完整的 hidden states
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info                                                 # forward_batch 获取 spec_info（包含初始 hidden states 和 top-k 信息）
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc                                         # forward_batch 获取 out_cache_loc（KV cache 位置）
        topk_p, topk_index, hidden_states = (                                               # 获取初始状态, 来自 forward_draft_extend 或上一轮的结果
            spec_info.topk_p,                                                               # top-k 概率值
            spec_info.topk_index,                                                           # top-k token IDs
            spec_info.hidden_states,                                                        # 最后位置的 hidden state
        )
        if self.hot_token_id is not None:                                                   # 如果使用了热词表限制，将索引映射回实际词表
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(                                              # kv cache 位置重排: [batch, topk, steps]
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(                           # 进一步重拍: [steps, batch, topk] -> [steps, -1] 
            self.speculative_num_steps, -1                                                  # 按时间步组织，循环中 out_cache_loc[i] 即可取当前步的位置
        )

        # Return values, 每步的树形结构信息
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):                                         # 循环执行 forward
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(              # 从 top-k 中选择当前步的输入 token, 更新 hidden states
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])                                                 # token 分数（用于后续组织树结构）
            token_list.append(tree_info[1])                                                 # 选中的 token IDs
            parents_list.append(tree_info[2])                                               # 父节点索引（构建树形关系）

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:                                         # 最后一步不需要执行 forward, Extend 阶段已经预填充了 1 个 token
                break                                                                       # 这里只需要生成 steps - 1 个新 token, 最后一步直接 break，节省一次 forward

            # Set inputs
            forward_batch.input_ids = input_ids                                             # draft 当前步选中的 token IDs
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,                                                        # 来自 Target Model 的 full hidden states
        next_token_ids: torch.Tensor,                                                       # Target 生成的首个 token
        seq_lens_cpu: Optional[torch.Tensor],                                               # 序列长度（CPU 张量）
        mm_input_embeds: Optional[torch.Tensor] = None,                                     # 多模态输入（可选）
    ):
        """Run draft model extend. This API modifies the states of the batch.               # Extend 阶段中 Draft Model 的初始化步骤，使用 Target Model 的输出对齐 Draft 的状态

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
            
        Example:
        Input                                                           
        ├── hidden_states: [h0, h1, h2, h3, h4] (from Target)         
        ├── next_token_ids: [t5] (from Target)                        
        └── seq_lens_cpu: 5                                           
                                                                
             ↓                                                         
        EagleDraftInput                                                
        ├── hidden_states = [h0, h1, h2, h3, h4]                      
        ├── verified_id = [t5]                                        
        └── num_tokens_per_req = 1                                    
                                                                       
             ↓                                                         
        Draft Model (如 Qwen3_5ForCausalLMMTP)                         
        ├── 输入: concat([emb(t5), h4]) @ fc → hidden                
        ├── 过 1 层 transformer                                       
        └── 输出: logits_output (预测 t6 的分布)                      
                                                                       
             ↓                                                         
        capture_for_decode                                             
        ├── topk_p = [0.4, 0.3, 0.2, 0.1]  (top-4 概率)               
        ├── topk_index = [t6_a, t6_b, t6_c, t6_d]  (top-4 tokens)     
        └── hidden_states = h5 (Draft 的最终状态)                     
                                                                       
        结果: batch.spec_info 已更新，准备好进入 Decode 循环           
        """
        batch.spec_info = EagleDraftInput(                                                  # 封装 Draft Model 的输入信息, Draft Model 不看原始文本，而是看 Target Model 的 hidden states
            hidden_states=hidden_states,                                                    # Target 的 full hidden states
            verified_id=next_token_ids,                                                     # Target 验证过的首个 token
            num_tokens_per_req=1,                                                           # 每个请求 1 个 token（初始化）
            num_tokens_for_logprob_per_req=1,                                               # logprob 计算 1 个 token
        )
        batch.return_hidden_states = False                                                  # Extend 阶段不需要返回 hidden states（只需要 Draft 内部使用）
        batch.spec_info.prepare_for_extend(batch)                                           # 准备 extend 所需的 metadata（如位置编码、mask 等）
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST                        # Draft 只需捕获最后位置的 hidden state（用于生成下一个 token）, 作为对比, Target 用 FULL（需要给 Draft 提供完整信息）
        model_worker_batch = batch.get_model_worker_batch(                                  # ScheduleBatch → ModelWorkerBatch: 标准模型输入格式
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(                                              # ModelWorkerBatch → ForwardBatch: 添加 Draft Model 特定的运行信息
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False                                                # 不计算 token 的 log probabilities
        if mm_input_embeds is not None:                                                     # 多模态处理：如果有 mm_input_embeds（多模态嵌入），传递给 forward batch
            forward_batch.mm_input_embeds = mm_input_embeds
        # self.draft_model_runner: Draft Model 的执行器, 如 Qwen3_5ForCausalLMMTP
        # 输入包含 Target 的 hidden_states（通过 forward_batch.spec_info）
        # 输出 logits_output 包含 Draft 预测的 logits 和 hidden states
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output        
        if self.enable_nan_detection:
            detect_nan(logits_output)                                                       # 检查输出是否有 NaN（训练/调试时使用）
        assert isinstance(forward_batch.spec_info, EagleDraftInput)                         # 确保 spec_info 类型正确
        assert forward_batch.spec_info is batch.spec_info                                   # 确保 forward_batch 和 batch 共享同一个 spec_info 对象（原地修改）
        self.capture_for_decode(logits_output, forward_batch.spec_info)                     # 捕获 Draft 的输出，为 Decode 阶段做准备

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                self.draft_model_runner.attn_backend.init_forward_metadata(
                    forward_batch
                )
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        if self.enable_nan_detection:
            detect_nan(logits_output)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)    # topk_p: Top-k 概率值, topk_index: Top-k token IDs
        draft_input.hidden_states = logits_output.hidden_states                             # hidden_states: 最后位置的 hidden state

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc