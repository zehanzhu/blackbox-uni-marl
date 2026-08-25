# multi_agent_blackbox 深度分析

本文分析 `examples/multi_agent_blackbox` 以及其依赖的 Uni-Agent 多智能体训练链路，重点说明：

- 外部黑盒 Multi-Agent System（MAS）如何接入训练框架；
- 多个 agent role 如何映射到多个可训练 Policy；
- Gateway 如何完成 role-aware 路由和 token trajectory 记录；
- 团队奖励如何转换成多 Policy 的 PPO/GRPO 更新；
- 当前实现的能力边界、正确性风险和生产化缺口。

## 1. 结论

当前实现的准确定位是：

> 单个外部黑盒 MAS + 多个 agent role + 多个可训练 Policy + 共享团队奖励 + 分 Policy PPO/GRPO 更新。

它采用“集中采样和团队 credit assignment，分 Policy 参数更新”的方式。外部 MAS 的编排逻辑可以保持在 Uni-Agent 之外，但所有需要训练的模型请求必须发送到 Uni-Agent Gateway，否则框架无法获得对应 Policy 的 token、logprob 和 trajectory。

当前示例不是“多个完全独立的 MAS 同时训练”，也不是 fully-async 多 Policy trainer。它是一个全局 MAS runner、一个全局 role-policy mapping，以及多个独立的同步 verl v1 PPO trainer。

## 2. 代码结构

### 2.1 示例层

| 文件 | 作用 |
| --- | --- |
| `examples/multi_agent_blackbox/config/multi_agent_blackbox.yaml` | 全局训练配置、role-policy 映射、每个 Policy 的 PPO 覆盖配置 |
| `examples/multi_agent_blackbox/config/mas_config.yaml` | 外部 MAS role 的 system prompt 和运行参数 |
| `examples/multi_agent_blackbox/multi_agent_runner.py` | 外部 MAS 的最小 HTTP runner |
| `examples/multi_agent_blackbox/reward.py` | 示例 reward function，生产环境应替换 |
| `examples/multi_agent_blackbox/verl_patch.py` | 多 Policy 共用 Ray 集群时的资源和 actor 名称补丁 |
| `examples/multi_agent_blackbox/scripts/run_train.sh` | 环境变量到 Hydra 配置的启动脚本 |

### 2.2 训练编排层

| 文件 | 作用 |
| --- | --- |
| `uni_agent/trainer/main_multi_agents_ppo.py` | Ray task runner、TransferQueue 生命周期和多 Policy 入口 |
| `uni_agent/trainer/multi_agents_ppo_trainer.py` | 创建多个 Policy trainer、采样、分组、优势计算和更新 |
| `uni_agent/trainer/framework/framework.py` | MultiAgent rollout 生命周期、reward 桥接和 TQ 写入 |
| `uni_agent/trainer/framework/entry.py` | 创建 Gateway runtime 和 trainer-facing rollout adapter |
| `uni_agent/trainer/gateway/gateway.py` | HTTP API、tokenization、generation、trajectory 记录 |
| `uni_agent/trainer/gateway/manager.py` | 在 Gateway actor 之间分配 rollout |
| `uni_agent/trainer/gateway/runtime.py` | 多 Policy LLM client 路由和 Gateway actor pool |

## 3. 总体架构

```text
dataset prompt
      |
      v
MultiAgentsPPOTrainer
      |
      +-- policy_1 PPOTrainer -- model_1 / tokenizer_1 / vLLM_1
      |
      +-- policy_2 PPOTrainer -- model_2 / tokenizer_2 / vLLM_2
      |
      +-- PolicyRoutingLLMClient
                |
                v
        Gateway actor pool
                |
                v
        rollout-scoped Gateway URL
                |
                v
        external blackbox MAS runner
          model=agent_1 / agent_2 / agent_3
                |
                v
        role -> policy -> LLM server
                |
                v
        role trajectories + rollout reward
                |
                v
        TransferQueue
                |
                +-- policy_1 batch -> PPO update
                |
                +-- policy_2 batch -> PPO update
```

核心拓扑由配置中的 `role_policy_mapping` 决定。例如：

```yaml
role_policy_mapping:
  agent_1: policy_1
  agent_2: policy_1
  agent_3: policy_2
```

这里的 trainer 数量由 `unique(policy_name)` 决定，而不是由 role 数量决定。因此 `agent_1` 和 `agent_2` 会共享 `policy_1` 的模型和优化器，但拥有独立的 Gateway session 和 trajectory。

## 4. 启动和初始化流程

入口是 `python -m uni_agent.trainer.main_multi_agents_ppo`，具体调用位于 `uni_agent/trainer/main_multi_agents_ppo.py`。

主要步骤如下：

1. Hydra 读取 `multi_agent_blackbox.yaml`。
2. Ray 初始化，并强制打开 TransferQueue。
3. 根据 `example_patch_fqn` 动态加载 `verl_patch.apply_patch()`。
4. 构造 `MultiAgentsPPOTrainer`。
5. 对每个 `policies.<policy_name>` 独立 compose 一份 verl `ppo_trainer` 配置。
6. 为每个 Policy 创建一个 verl v1 trainer。
7. 分别调用每个 Policy trainer 的 `init()`，创建 actor、rollout、tokenizer、reward worker、checkpoint engine 和 replay buffer。
8. 选择第一个 Policy 的 dataloader 和 replay buffer 作为外层共享数据源。
9. 创建一个 `PolicyRoutingLLMClient`，内部保存 `policy_name -> LLMServerClient`。
10. 创建一个共享的 `AgentFrameworkRolloutAdapter` 和 Gateway actor pool。
11. 进入外层 `MultiAgentsPPOTrainer.fit()` 循环。

值得注意的是，当前多 Policy trainer 明确拒绝 `colocate_async` 和 `separate_async`，见 `multi_agents_ppo_trainer.py` 中的 `_KNOWN_ASYNC_TRAINER_MODES` 检查。因此本示例是多 Policy 同步训练。

## 5. 外部黑盒 MAS 接口

示例 runner 位于 `examples/multi_agent_blackbox/multi_agent_runner.py`。

它收到以下关键输入：

- `raw_prompt`：数据集中的原始 prompt；
- `rollout`：当前 rollout 的 Gateway URL 和 role session；
- `role_policy_mapping`：role 到 Policy 的映射；
- `mas_config`：外部 MAS 的 role 配置。

runner 的行为是：

1. 从 `raw_prompt` 提取用户任务。
2. 为每个 role 创建 system message 和 user message。
3. 按 `role_policy_mapping` 的插入顺序依次调用每个 role。
4. 每次调用都向同一个 rollout-scoped Gateway URL 发送请求。
5. HTTP payload 中使用 `model=<role>`，而不是 `model=<policy_name>`。
6. 将之前 role 的文本输出追加到后续 role 的 user message 中。
7. 最后一个 role 的输出作为 `final_result`。
8. 返回 `agent_outputs` 和 `reward_info`。

Gateway 通过请求中的 role 找到对应的 `RoleSessionInfo`，再得到其 `policy_name`。因此外部 MAS 只需要知道 role 名和 OpenAI-compatible endpoint，不需要直接知道具体模型路径或 Ray actor。

当前 runner 是示意级实现：固定顺序、单轮、无重试、无退避、无并行调度，也没有真正执行 tool call。它只读取 `message.content`，如果模型只返回 `tool_calls` 而没有文本 content，角色输出会变成空字符串。

## 6. Gateway 和 trajectory 生命周期

### 6.1 Rollout 和 role session

一次 MAS rollout 会创建：

- 一个 `rollout_id`；
- 一个 rollout-scoped `base_url`；
- 每个 role 一个独立的 Gateway session；
- 每个 role 的 `role`、`session_id`、`policy_name` 元数据。

所有 role 请求都发送到同一个 URL，`model` 字段负责选择 role。这样做可以把一次 MAS 执行视为一个训练样本组，同时保留每个 role 的私有对话历史。

一次 role invocation 不是 trajectory finalize 边界。对于多轮或层级 MAS，同一个 role session 可以连续收到多次请求。只有 `finalize_multi_agent_rollout()` 才会物化该 rollout 中所有 role session 的 active trajectory。

### 6.2 Token 记录

Gateway 首次请求会：

1. 使用对应 Policy 的 tokenizer/processor 应用 chat template；
2. 构造完整 `prompt_ids`；
3. 调用对应 Policy 的 LLM server；
4. 将生成 token 加入 `response_ids`，并将生成 token 的 mask 设置为 1。

后续请求如果消息历史是已知历史的 prefix，Gateway 会：

- 复制当前 trajectory buffer；
- 对新增上下文 token 使用 `response_mask=0`；
- 对新生成 token 使用 `response_mask=1`；
- 将多轮生成合并到同一条 trajectory。

如果请求上下文不再是 prefix，Gateway 会先 materialize 旧 trajectory，再为新上下文创建新的 trajectory。

最终 trajectory 包含：

- `prompt_ids`；
- `response_ids`；
- `response_mask`；
- `response_logprobs`；
- `num_turns`；
- `multi_modal_data`；
- `role`；
- `policy_name`；
- `rollout_id`；
- `role_session_id`；
- reward 和其他 `extra_fields`。

当前 Gateway 默认请求 logprobs，见 `gateway.py` 中的 `_build_sampling_params()`。因此正常路径下 `rollout_log_probs` 会被写入 TQ，而不是使用全零占位值。

## 7. 从 rollout 到 TransferQueue

每个输入 prompt 会生成 `n` 个 MAS rollout。示例中 `n=4`。

对于一个 rollout，示例有三个 role，所以通常会产生三条 role trajectory：

```text
<uid>_0_0 -> agent_1 -> policy_1
<uid>_0_1 -> agent_2 -> policy_1
<uid>_0_2 -> agent_3 -> policy_2
```

实际 key 格式是：

```text
{uid}_{sample_idx}_{record_idx}
```

其中：

- `uid` 表示数据集 prompt 组；
- `sample_idx` 表示同一 prompt 下的第几个 MAS rollout；
- `record_idx` 表示该 rollout 内的 trajectory 序号。

每条记录的 TQ tag 会保存：

- `uid`；
- `rollout_id`；
- `sample_idx`；
- `record_idx`；
- `role`；
- `policy_name`；
- prompt/response/sequence 长度；
- global step 和 status。

ReplayBuffer 以 prompt UID 为单位判断一个 GRPO group 是否完成。即使一个 prompt 的部分 rollout 失败，只要至少有一个成功 rollout，当前逻辑仍可能将该 prompt 标记为可采样。

## 8. Reward 语义

示例 runner 返回的是 MAS 结果信息，不是最终数值 reward：

```python
{
    "final_result": final_result,
    "agent_outputs": agent_outputs,
}
```

之后 `MultiAgentFramework` 有两种 reward 路径：

1. 如果 runner 的 `reward_info` 中已经有 `score`、`reward` 或 `reward_score`，直接使用该值。
2. 如果没有数值 reward，则把最终 trajectory 和 runner 返回的 `reward_info` 交给 verl reward worker。

示例的 `reward.py` 会从 `extra_info.final_result` 中取最终 MAS 结果，并执行：

- `ground_truth` 存在时，检查期望文本是否为 final result 的 substring；
- 没有 ground truth 时，只要 final result 非空就给 1.0。

这个 reward 只是占位实现。生产任务应替换为测试通过率、规则检查器、环境执行结果或 judge model。

当前 reward 是 rollout-level team reward。一个 rollout 的 reward 会广播给其中所有 role trajectory，因此 agent_1、agent_2、agent_3 共享同一个最终任务信号。

## 9. 多 Policy PPO/GRPO 更新

外层 step 位于 `uni_agent/trainer/multi_agents_ppo_trainer.py` 的 `_step_once()`。

### 9.1 采样

每个 step 首先从第一个 Policy trainer 的 dataloader 取一批 prompt，为每个 prompt 生成新的 UID，然后提交到 AgentFramework。

外层 trainer 只使用一个共享 prompt 流。其他 Policy trainer 初始化时产生的 dataloader 和 dataset 会被释放。

### 9.2 按 Policy 分组

ReplayBuffer 返回的是包含所有 role trajectory 的混合 batch。trainer 遍历每条 TQ tag，根据 `policy_name` 分组：

```text
multi_agent_batch
      |
      +-- policy_1: agent_1 + agent_2 records
      |
      +-- policy_2: agent_3 records
```

这一步只按照 Policy 分组，不按 role 创建 trainer。因此共享同一个 Policy 的多个 role 会进入同一个 optimizer 更新。

### 9.3 Policy-specific old logprob

每个 Policy batch 分别执行：

1. sequence balancing；
2. `_compute_old_log_prob()`；
3. 可选 `_compute_ref_log_prob()`；
4. 可选 `_compute_values()`。

这里必须分别使用各自 Policy 的模型、tokenizer、temperature 和 actor worker，因为 `policy_1` 和 `policy_2` 可能是完全不同的模型。

### 9.4 共享团队 advantage

然后 trainer 将各 Policy batch 重新合并，根据所有 Policy 的 key 计算一次 advantage。

当前使用的 verl v1 GRPO helper 会：

1. 对每个 `{uid}_{sample_idx}` 找到最后一条 trajectory；
2. 使用同一 prompt 下不同 `sample_idx` 的最终 rollout reward 计算 GRPO relative advantage；
3. 将该 rollout advantage 广播到该 rollout 的所有 trajectory；
4. 将 advantage 和 returns 写回共享 TransferQueue。

因此对于 prompt `uid=u`、rollout `k`、role `r`：

```text
advantage(u, k, agent_1) = advantage(u, k)
advantage(u, k, agent_2) = advantage(u, k)
advantage(u, k, agent_3) = advantage(u, k)
```

### 9.5 分 Policy 更新 actor

最后每个 Policy trainer 只读取自己 TQ key 对应的 advantage、old logprob、response mask 和 response token，并执行自己的 critic/actor 更新。

最终效果是：

- reward 和 advantage 在 MAS rollout 层共享；
- actor 参数在 Policy 层隔离；
- role 是行为身份；
- policy 是真正的参数和资源隔离单位。

## 10. 资源隔离和 Ray 补丁

每个 Policy 的配置可以独立设置：

- model path；
- `nnodes`；
- `n_gpus_per_node`；
- FSDP size；
- tensor model parallel size；
- GPU memory utilization；
- checkpoint directory；
- vLLM rollout 参数。

verl 原有代码在多个 trainer 共用 Ray 集群时可能复用全局 placement group、reward worker 和 vLLM actor 名称。`verl_patch.py` 通过 monkey patch 给这些名称加入 Policy label，例如：

```text
policy_1_<random>verl_group_...
reward_loop_worker_policy_1_0
vllm_policy_1_server_0_0
```

这样两个 Policy 的 Ray 资源可以同时存在，并且不同 Policy 的 GPU placement group 不会因为同名而冲突。

该补丁依赖 verl 的内部类和方法名称，升级 Ray 或 verl 后需要重新执行资源验证脚本：

```text
examples/multi_agent_blackbox/scripts/verify_ray_pools.py
examples/multi_agent_blackbox/scripts/verify_vllm_servers.py
```

## 11. 当前实现的主要风险

### 11.1 高优先级正确性风险

#### 温度配置和实际采样可能不一致

配置在每个 Policy 下声明了 `rollout.temperature`，但示例 runner 没有把 temperature 放入 HTTP payload，Gateway 的 base sampling params 也为空。

当前值都是 `1.0`，所以实际默认采样和 old logprob 重算暂时一致。但如果将 Policy temperature 改成不同值，实际 vLLM 请求可能仍使用 SamplingParams 默认温度 `1.0`，而训练侧按配置温度重算 old logprob，导致 importance ratio 偏移。

如果要支持不同 Policy 或不同 role 的温度，必须让实际采样参数和训练侧重算参数拥有同一个来源，或者把每次采样的实际 temperature 写入 trajectory/TQ，并在 old logprob 阶段按样本使用。

#### 不同 Policy 的算法配置没有真正独立

虽然配置结构允许每个 Policy 写不同 PPO overrides，但 advantage 由第一个 Policy trainer 计算。以下配置应保持一致，除非扩展 trainer：

- `adv_estimator`；
- `rollout.n`；
- `gamma`；
- `lam`；
- `use_kl_in_reward`；
- GRPO normalization 相关配置。

否则最终结果依赖 `policies` 的插入顺序，而不是每个 Policy 自己的配置。

#### TransferQueue trajectory 没有在 update 后清理

verl 原生训练循环会在 batch 更新后清理 TQ key。当前外层 `MultiAgentsPPOTrainer.fit()` 没有对应的 `tq.kv_clear(keys=batch.keys)`，ReplayBuffer 只会清理 prompt UID，已消费的 trajectory 记录可能持续占用存储。

短实验可能不会暴露问题，长训练或较大 `n`、较多 role 时会导致 TransferQueue 容量持续下降。

### 11.2 中优先级工程问题

- 外层 trainer 没有真正执行 validation，`val_files` 和 `val_batch_size` 主要是为了满足 Policy trainer 初始化。
- `train_step()` 返回的 metrics 没有接入 verl Tracking/logger。
- `save_freq: -1` 且 `resume_mode: disable`，默认不会保存多 Policy checkpoint。
- `cleanup()` 方法存在，但主入口没有明确调用 trainer cleanup 或 Gateway shutdown。
- `run_train.sh` 可以修改 GPU 和 TP，但没有同步修改硬编码的 `fsdp_size: 8`，小规模配置可能不一致。
- role-policy mapping 没有在启动阶段完整验证，未知 Policy 可能在 rollout 或 batch 构建阶段才暴露。
- 每个 Policy 都会初始化 reward worker 和数据相关对象，但实际 reward handle 只取第一个可用 Policy 的 worker，存在资源浪费。

### 11.3 示例 runner 的能力限制

- 固定顺序调用 role；
- 没有并行 role execution；
- 没有 HTTP retry/backoff；
- 任一角色请求失败会 abort 当前 rollout；
- 不执行工具调用；
- 不支持复杂的层级 MAS 或内部 ReAct loop；
- 最终 role 由 mapping 的最后一个 role 决定；
- 没有 role-level reward 或 credit assignment。

## 12. 能力边界

当前实现适合：

- planner、executor、verifier 等固定 role 的 MAS；
- 多个 role 共享一个 Policy 的场景；
- 不同 role 使用不同模型或不同 tokenizer 的场景；
- 最终任务只有一个团队 reward 的场景；
- 同一数据集 prompt 下进行多次 MAS rollout 和 GRPO 比较的场景。

当前实现不直接保证：

- 每个 role 使用不同算法或不同 GRPO group size；
- 每个 role 使用不同 reward；
- 多个不同 MAS 拓扑在同一训练任务中自然切换；
- fully-async 多 Policy 更新；
- 训练 trajectory 的长期 TQ 存储回收；
- 生产级工具调用、失败重试和可观测性。

如果要支持多个 MAS 系统，当前通常需要让 runner 根据样本的 `agent_name` 或 `tools_kwargs` 分支，并使用一个可以覆盖全部 role 的全局 mapping。更完整的方案应为每个 MAS 配置独立的 role manifest、reward policy 和 rollout topology。

## 13. 建议的改进顺序

1. 统一实际采样 temperature 与 old logprob temperature，并增加自动化一致性测试。
2. 在每个外层 PPO update 完成后清理已消费的 TQ trajectory keys。
3. 在启动阶段验证 role-policy mapping、Policy 算法关键字段和资源配置的一致性。
4. 接入外层 metrics、validation、checkpoint 和 cleanup 生命周期。
5. 将 reward function 替换为真实任务 evaluator，并明确 team reward 与 role reward 的语义。
6. 为外部 runner 增加 tool call、retry、timeout、失败降级和并发控制。
7. 如果需要真正的异构 Policy 算法，按 Policy 分别计算 advantage，并设计跨 Policy 的 group/reward contract。

## 14. 测试和验证状态

本次分析执行了以下聚焦测试：

```text
76 passed, 1 skipped
```

跳过项是实际 verl/Ray v1 runtime 集成测试，当前环境缺少 `ray._raylet`。因此已验证的是配置、runner、Gateway、TQ metadata、batch 分组、advantage dispatch、checkpoint contract 等逻辑测试；尚未在本环境完成真实多 GPU、双 vLLM server 和完整训练 step 验证。

工作区在整理文档前无项目代码改动。

## 15. 重要源码定位

- `examples/multi_agent_blackbox/config/multi_agent_blackbox.yaml`
- `examples/multi_agent_blackbox/multi_agent_runner.py`
- `examples/multi_agent_blackbox/reward.py`
- `examples/multi_agent_blackbox/verl_patch.py`
- `uni_agent/trainer/main_multi_agents_ppo.py`
- `uni_agent/trainer/multi_agents_ppo_trainer.py`
- `uni_agent/trainer/framework/framework.py`
- `uni_agent/trainer/framework/entry.py`
- `uni_agent/trainer/gateway/gateway.py`
- `uni_agent/trainer/gateway/manager.py`
- `uni_agent/trainer/gateway/runtime.py`
- `tests/test_multi_agent_blackbox_example.py`
- `tests/uni_agent/trainer/test_multi_agent_gateway.py`
- `tests/uni_agent/trainer/test_multi_agents_ppo_trainer.py`

`examples/multi_agent_blackbox/to_be_solve.md` 是历史分析记录，部分内容对应旧版本代码。特别是温度、ReplayBuffer refill 和 logprob 相关结论，应以当前源码和实际运行验证为准。
