# RFC：多智能体黑盒训练

状态：草案

作者：Uni-Agent contributors

## 摘要

本 RFC 提议在 Uni-Agent 中新增一套原生的多智能体黑盒训练能力。该设计在
`examples/swe_agent_blackbox` 已有的黑盒 agent 训练路径基础上扩展：从单个
agent loop 和单个可训练 policy，扩展到多个外部 agent role、多个可训练 policy、
role-aware session routing，以及 per-policy PPO updates。

该方案保留黑盒集成模型，并将其扩展到多智能体系统：

- 外部多智能体系统仍然作为独立应用运行；
- role-aware gateway endpoint 接收来自各个 role 的 OpenAI-compatible model
  requests；
- role requests 被路由到对应的 policy rollout service；
- trajectory 使用 Uni-Agent 现有的 session-level `Trajectory` 格式收集；
- reward 被桥接回 per-policy training batches。

目标是让多智能体黑盒训练成为 Uni-Agent 的一等能力，并基于 Uni-Agent 现有的
session runtime、gateway、sandbox、reward 和 `verl` integration 构建。

## 动机

很多 agent 系统并不是单一的 monolithic policy，而是由多个不同角色组成的
orchestration graph，例如：

- verifier、searcher、answerer；
- planner、executor、reviewer；
- manager、coder、tester；
- router、specialist agents、final responder。

这些 role 可能需要不同的训练信号、sampling 参数、context budget、model checkpoint
或 update schedule。如果所有 role 都调用同一个 model endpoint，单智能体黑盒 runner
也可以执行这类系统，但它无法把每个 role 显式建模为一个可训练 policy。

生产级多智能体系统通常还有另一层约束：它们内部可以是层级式双层循环系统，但在
Uni-Agent 训练层只暴露 `agent_1`、`agent_2`、`agent_3` 这类抽象 agent roles。
某个 agent 可以持有外层 orchestration loop，维护私有 messages 和全局状态，反复把任务
分派给其他 agents，审查它们的最终总结，并判断继续、重试或完成。其他 agents 可以持有
自己的私有内层 ReAct loop，在同一次 rollout 中跨多次 invocation 持续保留自己的
message history，并且只返回最终总结。因此，一次 role invocation 不应被视为 trajectory
finalization 边界；Uni-Agent 应该让 role sessions 一直保留到整个 multi-agent rollout
完成。

这张图只关注 MAS 推理过程，而不是训练组件。每个 agent LLM request 都携带该 agent
配置的 `base_url`、`model` 和私有 messages，使不同 agent roles 可以调用不同的
model endpoints，同时隔离各自内部历史和工具轨迹。

![层级式双层循环多智能体系统](multi_agent_two_loop_inference.svg)

Uni-Agent 已经具备黑盒 agent 训练的核心基础设施：

- OpenAI-compatible gateway sessions；
- sandbox-backed agent execution；
- 通过 `complete_session(reward_info)` 传播 reward；
- 用于 scalable reinforcement learning 的 `verl` integration；
- `examples/swe_agent_blackbox` 中的 blackbox sidecar pattern。

目前缺少的是一个多智能体训练层，用于表达：

- role 到 policy 的映射；
- per-role request routing；
- role-aware session 和 trajectory collection；
- per-policy batch construction；
- multi-policy validation 和 checkpointing；
- 面向生产 workload 的 per-policy trainer resource isolation；
- 在不把外部 MAS 改写成 Uni-Agent 内部 agent loop 的前提下执行 blackbox external
  MAS。

本 RFC 提议新增一条 Uni-Agent-native multi-agent blackbox training path，具体范围包括：

- 支持外部多智能体系统的黑盒训练；
- 保留非侵入式集成模型，使外部 MAS 可以继续作为 standalone application 运行；
- 支持任意显式 role-policy mapping，包括共享 policy、per-role policy 和 mixed mappings；
- 通过 Uni-Agent-compatible gateway sessions 收集 token-level trajectories，同时保留
  role 和 policy metadata；
- 将收集到的 trajectories 转换为 per-policy `DataProto` batches；
- 当 roles 需要独立 policies 时，为每个 unique policy 编排一个 `RayPPOTrainer`，并在配置
  需要时使用独立 resource pools；
- 尽可能复用 Uni-Agent 现有的 `AgentFramework`、Gateway 和 `verl` training stack；
- 在 multi-agent rollout level 支持 GRPO 等 grouped rollout algorithms；
- 让现有 single-agent blackbox recipe 成为该设计的特例。

## 背景

### 现有 Uni-Agent 黑盒训练

当前黑盒训练 recipe 会在 sandbox 内或旁路运行外部 agent。runner 创建一个
Uni-Agent gateway session，启动 blackbox agent，将 agent 指向 gateway，评估 reward，
并通过 `complete_session(reward_info)` 返回结果。

当 rollout 在概念上是单 agent 和单 policy 时，这已经足够。对于多智能体系统，gateway
需要更多结构：每个 model request 都必须携带 role identity，路由到对应 policy，并在
finalized Uni-Agent trajectory 上保留 role/policy metadata。

### 期望的多智能体黑盒模式

一个实用的 Uni-Agent-native multi-agent blackbox training path 应保留现有的非侵入式
集成模型：

1. MAS application 由配置好的 `MultiAgentRunner` 启动。
2. `MultiAgentRunner` 将一个 rollout-scoped Uni-Agent Gateway `base_url` 注入 MAS
   model config。
3. 每个 request 使用 role name 作为 requested model。
4. 持有该 rollout 的 Gateway actor 将 role name 映射到 policy rollout backend。
5. Gateway 记录 messages、prompt ids、response ids、logprobs 和 metadata。
6. trajectory collector 按 role session 收集 finalized gateway trajectories，并附加
   role/policy metadata。
7. multi-agent framework 通过现有 reward worker/provider 机制计算或导入 final reward
   和 optional per-role rewards。
8. batch builder 按 `policy_name` 聚合可训练 records，并为每个 policy 输出一个
   `DataProto` batch。

对于 GRPO 等 grouped algorithms，同一个 dataset prompt 应创建一组独立的 multi-agent
rollouts。每个 rollout 拥有自己的 role-aware gateway sessions，并默认产出一个 final
task result。同一个 rollout 内，所有参与的 agent roles 应调用同一个 rollout-scoped
gateway `base_url`；`model` 字段用于标识 role。

本 RFC 提议使用 Uni-Agent-native abstractions 和 integration points 实现该模式。
role-aware request interception 应作为现有 Uni-Agent Gateway 的扩展来提供。

## 设计总览

该设计保留 Uni-Agent 现有 blackbox training path 的高层边界：trainer 通过
`AgentFramework.generate_sequences(prompts)` contract 请求 rollout batches，framework
implementation 启动外部 agent system，serving runtime 负责 gateway sessions 和
trajectory assembly。multi-agent extension 在此基础上新增 role-aware routing、
role-policy mapping、grouped rollout metadata 和 per-policy batch construction。

最接近的实现模型是当前的 `OpenAICompatibleAgentFramework`：创建 gateway sessions，启动
external runner，finalize Gateway-collected `Trajectory` objects，通过 reward worker 或
reward provider 为 finalized outputs 打分，并写出 Uni-Agent-compatible training records。
multi-agent proposal 将该生命周期从“一个 rollout sample 等于一个 gateway session”泛化为
“一个 rollout sample 是一组 role-aware gateway sessions”。

![Uni-Agent multi-agent training design overview](multi_agent_design_overview.svg)

## 提议架构

该架构引入一个 Uni-Agent 现有 `AgentFramework` contract 的 multi-agent blackbox
implementation，包含以下组件。
组件顺序从训练编排器和 policy/resource topology 开始，然后沿 rollout、gateway、trajectory、
reward 和 batch-building 路径展开。

### MultiAgentsPPOTrainer 组件

`MultiAgentsPPOTrainer` 是用于 multi-agent blackbox training 的 PPO trainer
orchestrator。它拥有 policy trainer registry，并协调一个或多个 `verl`
`RayPPOTrainer` instances。

它负责 top-level training loop：`MultiAgentsPPOTrainer.fit()` 收集 multi-agent
rollouts，计算 rollout-level training signals，并在已注册的 `RayPPOTrainer` instances 上
调度 per-policy update steps。

它负责：

- 校验配置中的 role-to-policy mapping；
- 为每个 unique trainable policy 创建一个 `RayPPOTrainer`；
- 将每个 policy trainer 分配到配置好的 `ray_resource_pool`；
- 基于 `uid` 分组的 reward tables 计算 rollout-level grouped advantages；
- 将每个 `rollout_id -> advantage` 附加到该 rollout 的所有 records；
- 将每个已附加 advantage 的 `policy_name -> DataProto` batch 分发给匹配的 trainer；
- 协调 skipped updates、global step consistency 和 per-policy checkpointing。

`MultiAgentsPPOTrainer` 调度 per-policy update steps，并调用每个已注册 `RayPPOTrainer`
上必要的 update interfaces。具体 PPO update mechanics 由底层 `RayPPOTrainer` instances
及其 worker groups 执行。multi-agent trainer 应将 rollout collection、batch construction、
grouped advantage assignment 和 per-policy update dispatch 保持为相互独立的职责。

#### Per-Policy PPO Runtime 配置

每个 policy 的 PPO runtime config 应在该 policy 的 config root 上独立 compose，然后再传给
该 policy 的 `RayPPOTrainer`。外层 multi-agent config 不应把完整 trainer config 直接
package 到 `policies.<policy_name>` 下面，因为公开 PPO trainer configs 中包含 root-level
Hydra defaults 和 root-relative interpolations。

推荐的配置形态是：

```yaml
ppo_trainer_config_source:
  kind: config_module
  value: verl.trainer.config

policies:
  policy_1:
    ppo_trainer_config_name: ppo_trainer
    ppo_trainer_overrides:
      trainer:
        ray_resource_pool: policy_1_pool
      actor_rollout_ref:
        model:
          path: ${oc.env:POLICY_1_MODEL_PATH,???}

  policy_2:
    ppo_trainer_config_name: ppo_trainer
    ppo_trainer_overrides:
      trainer:
        ray_resource_pool: policy_2_pool
      actor_rollout_ref:
        model:
          path: ${oc.env:POLICY_2_MODEL_PATH,???}
```

对每个 policy，`MultiAgentsPPOTrainer` 应：

- 从 `ppo_trainer_config_source` 将 `ppo_trainer_config_name` 作为 root-level config
  compose 出来；
- 在外层 multi-agent config 语境中 resolve `ppo_trainer_overrides`；
- 将 resolved overrides merge 到该 policy root config；
- 设置 policy identity，然后构造对应的 `RayPPOTrainer`。

这样可以保留公开 PPO trainer config 的原生语义，同时把 `role_policy_mapping`、
rollout-level Gateway settings 和 grouped rollout metadata 等 multi-agent orchestration
字段保留在外层 config。

### Policy-Trainer 拓扑与资源隔离

trainer topology 是 policy-centric 的。`role_policy_mapping` 首先将 `m` 个 MAS roles
解析到 `n` 个 unique trainable policies，其中 `n <= m`。`MultiAgentsPPOTrainer` 应根据
unique policies 集合推导 runtime trainer 和 resource topology，而不是根据 role 数量创建
trainers。

设计中应显式区分四个身份：

- **role**：外部 MAS agent identity，例如 `agent_1`、`agent_2` 或 `agent_3`；
- **policy**：Uni-Agent 和 `verl` 使用的可训练 model identity，例如 `policy_1`；
- **trainer**：per-policy `RayPPOTrainer` runtime 和 worker owner，执行被调度的 update
  steps；
- **ray_resource_pool**：分配给该 trainer 的 Ray resource pool。

核心 topology 是：

```text
role -> policy
unique(policy) -> RayPPOTrainer(policy) -> ray_resource_pool
```

因为 runtime trainers 是从 `unique(policy)` 推导出来的，所以该 topology 支持 one-to-one
mappings、all-shared mappings 和 mixed mappings。例如，`agent_1` 和 `agent_2` 可以共享
`policy_1`，同时 `agent_3` 映射到 `policy_2`：

```text
agent_1 -> policy_1 -> RayPPOTrainer(policy_1) -> ray_resource_pool_1
agent_2 -> policy_1 -> RayPPOTrainer(policy_1) -> ray_resource_pool_1
agent_3 -> policy_2 -> RayPPOTrainer(policy_2) -> ray_resource_pool_2
```

每个 unique policy 可以独立拥有 actor workers、rollout workers、tokenizer handles、
rollout server handles、optimizer state、checkpoint directories 和 failure domains。映射到
同一 policy 的 roles 共享该 policy 的 trainer 和 resource pool。大型 planner policy 因此可以
使用与较小 worker policy 不同的 GPU allocation、parallelism strategy、rollout capacity 或
checkpoint schedule，而有意共享 policy 的 roles 会复用同一套 runtime。

`MultiAgentsPPOTrainer` 应拥有 policy trainer registry，并强制维护该 topology：

- 为每个 unique policy 创建一个 `RayPPOTrainer`；
- 将每个 policy trainer 分配到配置好的 `ray_resource_pool`；
- 收集 per-policy rollout server、tokenizer 和 checkpoint handles；
- 使用 `role -> policy_name` metadata 支持 gateway routing 和 per-policy rollout backend
  handle collection；
- 将每个 `policy_name -> DataProto` batch 分发给匹配的 trainer；
- 跳过当前 step 中没有 samples 的 policies；
- 每个 configured policy 独立 checkpoint 和 restore。

gateway 使用 `role -> policy_name` 将 generation requests 路由到正确的 rollout backend。
可选的 `served_model_name` 字段只作为 rollout server 的 observability name，用于 metrics
和服务展示。batch builder 输出 `policy_name -> DataProto`。`MultiAgentsPPOTrainer` 随后只
更新当前 step 中收到真实 batch 的 trainers。

### MultiAgentFramework 组件

`MultiAgentFramework` 是一个 Uni-Agent `AgentFramework` implementation，
用于 blackbox multi-agent rollout。它保留现有 trainer-facing
`generate_sequences(prompts)` contract，同时在内部协调多个 role-aware gateway sessions、
external MAS execution、reward assignment、grouped reward metadata propagation 和
per-policy batch construction。

它负责：

- 将每个 prompt 展开为 `rollout.n` 个 multi-agent rollout samples；
- 为每个 rollout 创建一个 rollout-scoped gateway `base_url`，并为被调用的 role 创建
  role-aware gateway sessions；
- 调用配置好的 blackbox MAS runner；
- 收集 role-aware gateway session trajectories；
- 为每个 multi-agent rollout 打分；
- 分配 rollout-level 或 role-level training signals；
- 产出 per-policy batches 或 scored trajectories；
- 协调 validation 和 failure handling。

对于 single-agent use cases，该 framework 应能在只有一个 role 和一个 policy 的情况下运行。

### 与 OpenAICompatibleAgentFramework 的关系

`MultiAgentFramework` 应复用 `OpenAICompatibleAgentFramework` 的 lifecycle
pattern，并在现有 rollout stack 上扩展 multi-agent execution。

现有 single-agent blackbox lifecycle 是：

```text
uid
  -> rollout.n independent gateway sessions
  -> agent_runner(session.base_url)
  -> Gateway records Trajectory objects
  -> finalize_session(session_id)
  -> _score_trajectories(...)
  -> write TransferQueue-compatible training records
```

multi-agent lifecycle 应将该模式扩展为：

```text
uid
  -> rollout.n independent multi-agent rollouts
  -> each rollout owns one rollout-scoped gateway base_url
  -> each rollout owns multiple role-aware gateway sessions behind that base_url
  -> MultiAgentRunner launches the external MAS
  -> MAS repeatedly invokes that base_url with model=<role>
  -> each role session remains ACTIVE across repeated role invocations
  -> Gateway records one or more Trajectory objects per role session
  -> MultiAgentRunner finishes the whole multi-agent rollout
  -> framework runs finalize_multi_agent_rollout(rollout_id)
  -> score the finalized multi-agent rollout
  -> attach rollout-level reward and grouped metadata
  -> emit per-policy training records
```

这样可以保持 trainer integration 稳定：trainer 仍然调用 `generate_sequences(prompts)`，
multi-agent framework 负责多个 roles、多个 sessions、grouped rewards 和 per-policy
output 所需的额外协调。

一次 role invocation 不是 Gateway session 边界。在 hierarchical MAS workloads 中，
某个 agent 可能在一次 rollout 中多次调用同一个被调度的 agent role。该 agent 保持自己的私有
message history，append 新的 instruction，运行内部 ReAct loop，然后
只返回 final text result。role-aware Gateway session 必须在这些
invocations 之间保持 active。`finalize_session(session_id)` 只应在 rollout finalization
阶段作为 session-level primitive 使用，而不是作为 public multi-agent abstraction。framework
应在 `MultiAgentRunner` 完成整个 multi-agent execution 之后提供一个 rollout-level
finalization 步骤，概念上可以称为 `finalize_multi_agent_rollout(rollout_id)`。这是 public
multi-agent boundary。内部实现应由拥有该 `rollout_id` 的 Gateway actor 基于 rollout
manifest materialize 该 rollout group 内所有 role-session trajectories，并组装为一个
`MultiAgentRolloutResult`。

### RolePolicyMapping 组件

`RolePolicyMapping` 将外部 MAS role names 映射到可训练 policy names。

示例：

```yaml
roles:
  - agent_1
  - agent_2
  - agent_3

role_policy_mapping:
  agent_1: policy_1
  agent_2: policy_1
  agent_3: policy_2
```

第一版应支持从 configured roles 到 configured trainable policies 的任意显式映射。特例包括
one-to-one mappings、all-shared mappings，以及部分 roles 共享 policy 的 mixed mappings。

### RoleAwareGateway 组件

`RoleAwareGateway` 是 Uni-Agent 现有 Gateway 的扩展。它向外部 MAS 暴露相同的
OpenAI-compatible session endpoints，同时新增 role-aware routing 和 recording。MAS 仍然
正常调用 chat completions。requested model name 或另一个显式 request field 用于标识
agent role。

对于 multi-agent rollout，面向 Gateway 的 `base_url` 应是 rollout-scoped，而不是
role-specific。外部 MAS 应将同一个 rollout 内的所有 agent model clients
都指向同一个 gateway `base_url`，并在每个 request 中设置 `model=<role>`。Gateway actor
使用 request model 选择 role session 和 policy rollout backend。

gateway 负责：

- 校验 active role-aware gateway session；
- 提取 role；
- 使用该 role 的 tokenizer 将 messages 渲染成 prompt ids；
- 将 request 路由到正确的 policy rollout backend；
- 返回 OpenAI-compatible response；
- 用 Uni-Agent 现有 trajectory format 记录 generated tokens。

第一版实现应尽可能复用 `GatewayServingRuntime`、`GatewayManager` 和 `GatewayActor`
概念。对于 single-agent workloads，role-aware gateway session 应表现得像现有 blackbox
gateway session，只是在 multi-agent workloads 中额外携带 role 和 policy metadata。

### GatewayManager 与 Gateway 职责

`GatewayManager` 应继续作为 session-routing layer。它拥有由 `gateway_count` 创建的
Gateway actor pool，在创建 session 或 rollout session group 时选择一个 Gateway actor，并将
`complete_session`、`finalize_session`、`finalize_multi_agent_rollout` 和 `abort_session`
调用路由回拥有该 session group 的 actor。它不应拥有 tokenization、role routing、
rollout generation、reward handling 或 trajectory materialization。

对于 multi-agent rollout，`GatewayManager` 应保持 rollout-level actor affinity：属于同一个
`rollout_id` 的所有 role-aware sessions 都应由同一个 Gateway actor 持有。不同 rollout
session groups 仍然可以分布到不同 `gateway_count` actors 上做负载均衡。

Gateway actor 应拥有 role-aware training behavior。它接收发往 rollout-scoped gateway
`base_url` 的 OpenAI-compatible request，从 request `model` 中提取 role，将 role 解析为
policy，在该 rollout session group 内选择对应 role session state，使用 policy tokenizer
渲染 prompt，调用 role-aware rollout backend，并在 rollout finalize 时 materialize
trajectories。

这使得 multi-agent proposal 与当前 Uni-Agent gateway 分层保持一致：`GatewayManager`
负责在 Gateway actors 之间调度 sessions 或 rollout session groups，而每个 Gateway actor
负责自身 sessions 的 request handling 和 trajectory state。

### Rollout Session Group Manifest 组件

当前 Uni-Agent Gateway state 是 session-scoped：

```text
GatewayActor
  _sessions[session_id] -> GatewaySessionState
```

multi-agent rollout 需要在现有 session state 之上增加一层。Gateway actor 应维护一个
rollout-level manifest，用于把属于同一次 multi-agent execution 的 role sessions 组织在一起：

```text
MultiAgentRolloutState
  rollout_id
  sample_idx
  uid
  phase
  role_sessions:
    agent_1:
      session_id
      policy_name
      phase
      metadata
    agent_2:
      session_id
      policy_name
      phase
      metadata
    agent_3:
      session_id
      policy_name
      phase
      metadata
  metadata:
    parent_child_links
    invocation_records
    reward_info
```

该 manifest 是 routing 和 lifecycle index，而不是 `GatewaySessionState` 的替代品。每个
role session 应继续复用 Uni-Agent 现有的 `GatewaySessionState`、message history、
`TrajectoryBuffer`、full encoding 和 incremental encoding 行为。额外的 rollout layer
提供：

- `rollout_id -> role -> session_id` lookup；
- `session_id -> rollout_id`、role 和 policy metadata；
- completion、finalization 和 abort 所需的 rollout-level phase tracking；
- 同一个 rollout 内所有 role sessions 的 actor affinity；
- 记录 invocation relationships 和 optional runner metadata 的稳定位置。

`finalize_multi_agent_rollout(rollout_id)` 应路由到拥有该 rollout group 的 Gateway actor。
该 actor 应读取 manifest，materialize 每个底层 role session 的 active trajectory，将返回的
trajectories 组装为 `MultiAgentRolloutResult`，然后删除 manifest 和已 finalized 的 role
sessions。这样可以保持 trajectory storage 仍然是 Uni-Agent-native，同时为 multi-agent
training 提供一等的 rollout boundary。

### Multi-Agent Rollout API Contract 组件

public multi-agent lifecycle 应是 rollout-scoped。现有 session-level APIs 仍然适用于
single-agent blackbox workloads，而 multi-agent path 应暴露一组小的 rollout-level
contract：

```python
create_multi_agent_rollout(
    rollout_id: str,
    roles: list[str],
    role_policy_mapping: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> MultiAgentRolloutHandle

complete_multi_agent_rollout(
    rollout_id: str,
    reward_info: dict[str, Any] | None = None,
) -> None

wait_for_multi_agent_rollout_completion(
    rollout_id: str,
    timeout: float | None = None,
) -> None

finalize_multi_agent_rollout(
    rollout_id: str,
) -> MultiAgentRolloutResult

abort_multi_agent_rollout(
    rollout_id: str,
) -> None
```

`MultiAgentRolloutHandle` 应包含：

- `rollout_id`；
- 一个 rollout-scoped Gateway `base_url`；
- role 到 model name 的映射，默认 model name 可以直接使用 role name；
- 用于 debugging 和 trajectory attribution 的 role session metadata。

该 handle 有意只暴露一个属于整个 rollout 的 `base_url`。MAS 中所有 agent
requests 都使用同一个 `base_url`，并通过设置 `model=<role>` 选择 role。Gateway actor
随后通过 rollout manifest 将 `model=<role>` 解析到 role session、policy、tokenizer 和
rollout backend。

`create_multi_agent_rollout` 应是分配 rollout-level actor affinity 的唯一位置。`GatewayManager`
为该 `rollout_id` 选择一个 Gateway actor，该 rollout 下的所有 role sessions 都在该 actor
内创建。底层 role sessions 仍然可以由内部 `session_id` 和 `GatewaySessionState` 表示，但
除 logging、debugging 或 trajectory metadata 需要外，这些 session ids 应是实现细节。

`complete_multi_agent_rollout` 将 rollout group 标记为 completed，并存储 rollout-level
`reward_info` 或 runner metadata。它不应 materialize trajectories。

`wait_for_multi_agent_rollout_completion` 用于等待异步启动的 MAS 调用
`complete_multi_agent_rollout`。同步 runner 可以跳过这一步，在 runner 返回后直接
finalize。它对应 single-agent 的 `wait_for_completion(session_id)`，但保留 rollout-level
multi-agent boundary。

`finalize_multi_agent_rollout` 是 public finalization boundary。它应路由到 owning Gateway
actor，materialize rollout group 内所有 role-session trajectories，附加 rollout 和 role
metadata，组装 `MultiAgentRolloutResult`，并清理 rollout manifest 以及 role sessions。

`abort_multi_agent_rollout` aborts rollout group，并清理所有 role sessions。如果 rollout
已经 aborted 或已经被清理，该操作应是 idempotent。

现有 `create_session`、`complete_session`、`finalize_session` 和 `abort_session` APIs 仍然是
session-level primitives。multi-agent framework 不应要求 callers 为一个 rollout 中的每个
role 直接编排这些 session-level APIs。

### 增量编码兼容性

role-aware Gateway 应复用 Uni-Agent 现有的 full encoding 和 incremental encoding 路径。
session 中的第一个 request 可以被编码为 full prompt。后续 request 如果是在之前 message
history 基础上追加内容，应使用 incremental encoding，仅编码新增 messages，并保留现有
`response_mask` 语义：

- model-generated tokens 是可训练的，应有 `response_mask=1`；
- 新增 context tokens、tool results 或其他 non-generated tokens 应有 `response_mask=0`；
- finalized trajectories 应保留 prompt ids、response ids、response masks、logprobs、
  role name、policy name 和 session metadata。

这对 hierarchical multi-agent systems 很重要，因为每个 agent 可能拥有自己的私有 message
state，并可能在一次 rollout 中被多次调用。role-aware Gateway 应为每个 agent session
复用相同的 full/incremental encoding semantics，而不是引入单独的 multi-agent-specific
storage format。

framework 不应在某个 agent 完成一次内部 ReAct loop 时 finalize 该 role session。如果同一个
agent 在同一个 multi-agent rollout 后续再次被调用，新的 instruction 应 append
到该 agent 现有私有 message history 中，并通过同一个 session-level incremental path 编码。

### RoleAwareRolloutBackend 组件

`RoleAwareRolloutBackend` 将 role requests 路由到 policy rollout services。

输入：

- role name；
- external MAS request 中的 model override；
- prompt ids；
- sampling parameters。

输出：

- response text；
- response token ids；
- logprobs；
- finish reason；
- routing metadata。

backend 应尽早拒绝 unsupported generation parameters。对于第一版，每个 model request
使用 `n=1` 就足够；multiple samples 应在 episode level 处理。

### MultiAgentRunner 组件

`MultiAgentRunner` 启动外部 MAS。它类似 Uni-Agent 现有 blackbox agent runner pattern，
但必须向 MAS config 注入一个 rollout-scoped gateway `base_url` 和 role-specific model
names。

runner 应是一个支持多种 execution modes 的 launch abstraction。具体 runner 可以
in-process 执行 MAS，也可以在 dedicated Ray worker、sandbox、sidecar tool image 或
external CLI/subprocess 中执行。

可配置的 runner 可以从 `multi_agent_runner_kwargs.mas_config_path` 加载一个 MAS config
文件。该 config 应在 `agents` 下定义抽象 agent roles，包括每个 agent 的 prompt、model
role name、tool parser、tools，以及其他 MAS-specific runtime options：

```yaml
agents:
  agent_1:
    model: agent_1
    tool_parser: qwen3_coder
    tools:
      - name: spawn_agent
      - name: finish

  agent_2:
    model: agent_2
    tool_parser: qwen3_coder
    tools:
      - name: str_replace_editor
      - name: execute_bash

  agent_3:
    model: agent_3
    tool_parser: qwen3_coder
    tools:
      - name: search
      - name: submit
```

对于可配置的 MAS runner，framework 可以：

1. 加载或渲染 MAS config；
2. 将 global 和 per-agent model `base_url` 都替换为同一个 rollout-scoped gateway
   `base_url`；
3. 将每个 agent 的 model name 设置为 agent role name，使 `model=<role>` 选择 role
   session 和 policy rollout backend；
4. 通过配置的 runner mode 启动 MAS；
5. 按 episode 持久化 stdout 和 stderr logs；
6. 在完成后返回 final MAS result 和 rollout metadata。

runner 不应在某个 agent 完成一次内部 ReAct loop 时 finalize role session。runner completion
boundary 是整个 multi-agent rollout。到达该边界后，`MultiAgentFramework`
会运行一个 rollout-level finalization 步骤，概念上可以称为
`finalize_multi_agent_rollout(rollout_id)`，并构建 `MultiAgentRolloutResult`。第一版实现
应在拥有 `rollout_id` 的 Gateway actor 上 finalize rollout session group，而不是要求
framework 对每个 role session 发起 public `finalize_session` 调用。

这保持了与现有 blackbox recipe 的一致性，同时允许 multi-agent examples 根据自身环境选择合适
的 execution mode。

### MultiAgentTrajectoryCollector 组件

`MultiAgentTrajectoryCollector` 读取 rollout session group manifest，并收集属于同一个已完成
multi-agent rollout 的所有 role-aware gateway sessions 的 finalized Uni-Agent `Trajectory`
objects。

```text
MultiAgentRolloutResult
  rollout_id
  sessions:
    agent_1:
      session_id
      policy_name
      trajectories: [Trajectory, ...]
    agent_2:
      session_id
      policy_name
      trajectories: [Trajectory, ...]
    agent_3:
      session_id
      policy_name
      trajectories: [Trajectory, ...]
  metadata:
    uid
    sample_idx
    parent_child_links
    invocation_records
    reward_info
```

每个 agent 保持自己的 Gateway session 和私有 message history。这对于 hierarchical MAS
workloads 很重要：某个 agent 可以拥有全局 orchestration loop，而其他 agents 拥有内部
ReAct loop，并且只返回 final result。

这些 role sessions 应在整个 multi-agent rollout 期间保持 active。某个 agent 完成一次内部 ReAct
loop 只是一次 role invocation completion，不是 Gateway session completion。rollout
completion 时，framework 调用 `finalize_multi_agent_rollout(rollout_id)`，收集所有 role
sessions 的 finalized `Trajectory` objects，然后组装 rollout-level result。Gateway actor
可以复用 `finalize_session` 使用的 private materialization 逻辑，但 multi-agent lifecycle
应按 `rollout_id` finalize，而不是迭代 public per-session finalization calls。

collector 不应引入单独的 multi-agent-specific training data format。相反，它应保留当前
Uni-Agent `Trajectory` fields，并将 rollout manifest 中的 role/policy/session metadata 写入
rollout result，并在 batch construction 需要时写入 `Trajectory.extra_fields`：

- `rollout_id`；
- `uid`；
- optional `sample_idx`；
- `session_id`；
- role name；
- policy name；
- optional parent session id；
- optional invocation id；
- request metadata。

role-session 保留和 policy 分组是两个独立步骤。一个完成的 rollout 会先按 role session
保留 trajectories，便于追踪每个 agent 的私有 session、message history 和调用链；随后在
batch construction 阶段，再按 `policy_name` 重新分组。如果两个 roles 共享同一个 policy，
最终的 policy batch 应同时包含这两个 role sessions 的 records：

```text
rollout_result.sessions:
  agent_1: {session_id: s_agent_1, policy_name: policy_1}
  agent_2: {session_id: s_agent_2, policy_name: policy_1}
  agent_3: {session_id: s_agent_3, policy_name: policy_2}

per_policy_records:
  policy_1: [s_agent_1 trajectories, s_agent_2 trajectories]
  policy_2: [s_agent_3 trajectories]
```

### Reward 分配

Reward 分配应是 framework 内部逻辑，而不是新的 Gateway service 或独立组件。
`MultiAgentFramework` 应遵循当前 `OpenAICompatibleAgentFramework` 的模式：
finalized trajectories 被转换为 reward inputs，配置好的 reward worker 或 provider 计算
score，然后 framework 在 batch construction 前将 reward values 和 group metadata 附加到
可训练 trajectories。

runner 可以将 task-specific `reward_info` 和 final MAS result 一起返回。这与当前 blackbox
SWE-agent recipe 一致：runner 在环境中完成 evaluation，返回 `reward_info`，framework 再将
这些信息注入到 `RewardLoopWorker` 和配置的 `custom_reward_function` 消费的 reward input 中。

framework 应将这些数据存储在 rollout result 中，例如：

```python
{
    "final_result": {
        "answer": "...",
        "status": "completed",
    },
    "reward_info": {
        "reward_score": 1.0,
        "resolved": True,
    },
    "extra_info": {...}
}
```

对于 rollout-level rewarding，framework 应从 finalized `MultiAgentRolloutResult`、sample
metadata、final MAS result，以及 optional runner-provided `reward_info` 构造 reward input。
随后它将该输入派发给配置的 reward worker/provider。返回的 scalar score 就是 rollout
reward。第一版应支持 scalar rollout rewards 和 optional role-level reward metadata。
Token-level reward shaping 可以后续添加。

Gateway 不应计算 reward。它只应通过 `complete_session(reward_info)` 和 finalized
`Trajectory` objects 存储并传播 `reward_info`。reward computation 和 assignment 应继续位于
framework 层，遵循 `OpenAICompatibleAgentFramework` 的模式：finalized outputs 被转换为
reward inputs，reward worker 或 provider 计算 score，然后 framework 在 batch construction
前将 reward values 和 group metadata 附加到可训练 trajectories。

概念流程：

```text
MultiAgentRunner result
  -> final_result + optional reward_info
  -> finalize_multi_agent_rollout(rollout_id)
  -> MultiAgentRolloutResult
  -> build reward input from rollout result + sample metadata
  -> RewardLoopWorker/custom_reward_function computes reward_i
```

### MultiPolicyBatchBuilder 组件

`MultiPolicyBatchBuilder` 接收来自 `MultiAgentRolloutResult` 的 finalized 且 scored 的
Uni-Agent `Trajectory` objects，并输出 `policy_name -> DataProto`。

它不负责 finalize Gateway sessions，不负责计算 rewards，也不负责更新 policy trainers。
它的职责是在保留 Uni-Agent trajectory semantics 的同时，按 `policy_name` 重新组织可训练
records。

输入 contract 包括：

- 来自一个或多个已完成 `MultiAgentRolloutResult` 的 finalized role-session trajectories；
- `rollout_id`、`sample_idx` 等 rollout metadata；
- Uni-Agent 现有 `uid` 字段作为 grouped-rollout key；
- `session_id`、role name 和 `policy_name` 等 session metadata；
- framework 已经分配好的 reward scores 和 optional group metadata。

builder 应执行以下操作：

- 将 role-session trajectories flatten 为 trainable records；
- 保留现有 `Trajectory` fields，包括 `prompt_ids`、`response_ids`、`response_mask`、
  `response_logprobs`、`multi_modal_data`、`routed_experts` 和 `extra_fields`；
- 保持 Uni-Agent 现有 `response_mask` 语义不变，确保只有 model-generated tokens 可训练；
- 将 rollout-level rewards 附加到 trainable records；当所选算法需要 token-level reward
  placement 时，将 scalar reward 分配到最后一个有效 generated response token；
- 保留 `uid`、`rollout_id` 和 `sample_idx` 作为主要 grouped-rollout keys，并与 Uni-Agent
  现有 TransferQueue 和 GRPO grouping 语义对齐；
- 按 `policy_name` 对 records 分组，并为当前 step 中出现的每个 policy 构建一个
  `DataProto` batch。

当多个 role sessions 解析到同一个 `policy_name` 时，这些 records 应拼接进同一个
`DataProto`。当前 step 中没有可训练 records 的 policies 不应出现在输出中，由
`MultiAgentsPPOTrainer` 跳过它们的 update。

### Grouped Rollout 与 GRPO Advantage Assignment

对于 GRPO-style training，`rollout.n` 应为同一个 dataset prompt 创建多个独立的
multi-agent rollout samples。如果 `rollout.n=8`，framework 会启动八个
`MultiAgentRolloutResult` executions。每个 execution 拥有自己的一组 role-aware gateway
sessions，每个参与的 agent role 对应一个 session。

![Grouped multi-agent GRPO flow](multi_agent_grpo_flow.svg)

默认情况下，每个 multi-agent rollout 产出一个 final task result 和一个 scalar reward：

```text
uid
  rollout_0 -> final_result_0 -> reward_0
  rollout_1 -> final_result_1 -> reward_1
  ...
  rollout_7 -> final_result_7 -> reward_7
```

对于每个 rollout sample `i`，`reward_i` 由配置的 reward worker/provider 基于 finalized
`MultiAgentRolloutResult` 和 sample metadata 计算得到。如果 `MultiAgentRunner` 已经产出
task-specific `reward_info`，framework 应将其注入 reward input，遵循
`OpenAICompatibleAgentFramework` / SWE-agent pattern。

framework 应将 `reward_i` 和 grouping metadata 附加到 rollout `i` 内所有可训练 trajectory
records。随后，`MultiAgentsPPOTrainer` 应先基于整个 `uid` group 的 rollout reward table
计算 GRPO advantages，再分发 per-policy updates：

```text
uid
  rollout_0 -> reward_0
  rollout_1 -> reward_1
  ...
  rollout_7 -> reward_7

rollout_advantages = grpo_advantages([reward_0, reward_1, ..., reward_7])
rollout_0 -> advantage_0
rollout_1 -> advantage_1
...
rollout_7 -> advantage_7
```

rollout `i` 的 reward 应对 rollout `i` 内所有可训练 Uni-Agent `Trajectory` 可见，跨所有
roles 和 policies：

```text
rollout_i
  agent_1 session trajectories -> reward_i + group metadata
  agent_2 session trajectories -> reward_i + group metadata
  agent_3 session trajectories -> reward_i + group metadata
```

computed advantage 属于 PPO update pipeline，默认不应由 framework 预先 materialize。
完成 reward 和 group metadata assignment 后，`MultiPolicyBatchBuilder` 按 `policy_name`
对 trajectory records 分组，并输出 `policy_name -> DataProto`。`MultiAgentsPPOTrainer`
随后将共享的 `rollout_id -> advantage` table 应用到 per-policy batches 中的每条 record：

```text
policy_1 batch:
  records from rollout_0 -> advantage_0
  records from rollout_3 -> advantage_3

policy_2 batch:
  records from rollout_0 -> advantage_0
  records from rollout_1 -> advantage_1
```

只有完成 rollout-level advantage assignment 后，各 policy trainer 才运行自己的 PPO update。
因此，一个 policy batch 不是一个 role-session batch，而是同一 rollout group 中所有 metadata
解析到同一个 policy 的可训练 records 的集合。

这使第一版与 blackbox task-level rewarding 保持一致。更复杂的 role-level 或 policy-level
credit assignment 可以后续添加，而不需要改变 role-aware gateway session model。

## 多智能体 RL 训练流程

```text
dataset sample
  -> MultiAgentFramework
  -> create rollout.n independent multi-agent rollout samples
  -> create one rollout-scoped gateway base_url and role-aware sessions
  -> MultiAgentRunner launches external MAS
  -> MAS calls the rollout gateway base_url with model=<role>
  -> gateway actor extracts role, selects role session, and renders prompt ids
  -> RoleAwareRolloutBackend routes to policy rollout server
  -> gateway records Uni-Agent Trajectory data with role/policy metadata
  -> MAS may invoke the same role session multiple times
  -> runner exits after the whole multi-agent rollout
  -> framework runs finalize_multi_agent_rollout(rollout_id)
  -> trajectory collector preserves finalized trajectories by role session with role/policy metadata
  -> framework builds reward input from rollout result and reward_info
  -> reward worker/provider computes one reward per multi-agent rollout
  -> framework attaches rollout reward and group metadata to trainable trajectories
  -> MultiPolicyBatchBuilder emits per-policy DataProto with rewards and group metadata
  -> MultiAgentsPPOTrainer computes uid-grouped rollout advantages
  -> MultiAgentsPPOTrainer attaches rollout_id -> advantage to per-policy records
  -> each present policy updates its own trainer
```

上面的文字流程是端到端摘要。下面的时序图用组件交互和 public API boundary 的形式展示同一流程。

## 多智能体 RL 训练时序图

![多智能体 RL 训练时序图](multi_agent_training_sequence.svg)

## 实现计划

### Phase 1：Role-Aware Gateway 与 Runner

- 扩展 Uni-Agent Gateway，使其接受 role-aware OpenAI-compatible chat requests。
- 从每个 request 中提取 role，解析 `role -> policy_name`，并将 generation 路由到匹配的
  rollout backend。
- 复用 `OpenAICompatibleAgentFramework` session lifecycle 作为新的 multi-agent framework
  implementation 的基线。
- 在 Gateway actor 中定义 `MultiAgentRolloutState` 和 rollout session group manifest。
- 添加 rollout-level actor affinity，使同一个 `rollout_id` 下所有 role sessions 都分配到同一个
  Gateway actor。
- 通过 `MultiAgentRunner` 启动一个简单 external MAS。
- 在同一个 rollout 的重复 role invocations 之间保持 role-aware sessions active。
- 添加 rollout-level finalization 步骤，概念上可以称为
  `finalize_multi_agent_rollout(rollout_id)`，仅在整个 multi-agent rollout 完成后收集带
  role 和 policy metadata 的 finalized gateway trajectories。

### Phase 2：Trajectory 与 Reward 分配

- 定义 `MultiAgentRolloutResult`、agent session metadata 和 reward payload schemas。
- 添加 reward worker/provider invocation utilities。
- 支持 runner 产出的 `reward_info`，并按照现有 SWE-agent blackbox pattern 将其合并进
  reward input metadata。
- 将 Gateway reward behavior 限定为存储和传播 `reward_info`。
- 将 finalized gateway trajectories 转换为 per-policy rollout records。
- 支持 rollout-level reward payloads，同时保留 optional role-level reward metadata，用于后续
  credit-assignment extensions。

### Phase 3：Per-Policy Batch Building

- 将 finalized Uni-Agent trajectories 转换为 per-policy `DataProto` batches。
- 将 rollout-level rewards 传播到同一 `rollout_id` 下的所有 trainable records，同时保留
  optional role-level metadata，用于下游 credit-assignment extensions。
- 将 reward 分配到最后一个有效 generated response token。
- 对 grouped algorithms，保留 `uid`、`sample_idx` 和 `rollout_id`，使
  `MultiAgentsPPOTrainer` 可以在同一个 Uni-Agent prompt uid 的所有 samples 上计算
  rollout-level group-relative advantages。

### Phase 4：MultiAgentsPPOTrainer

- 添加 `MultiAgentsPPOTrainer` 作为 multi-policy PPO orchestration layer。
- 将 `role_policy_mapping` 解析为 unique trainable policies 集合。
- 对每个 policy，从 `ppo_trainer_config_name` 指定的公开 PPO trainer config 在 policy
  config root 上独立 compose PPO runtime config，然后应用 `ppo_trainer_overrides`。
- 添加 policy trainer registry，为每个 unique policy 创建一个 `RayPPOTrainer`。
- 将每个 unique policy trainer 分配到配置好的 resource pool。映射到同一 policy 的 roles
  共享该 trainer 和 resource pool。
- 为每个 policy 收集 rollout server handles、tokenizer handles 和 checkpoint managers。
- 添加 grouped advantage assigner，基于 `uid` 分组的 reward table 计算
  `rollout_id -> advantage`，并将每个 rollout advantage 附加到该 rollout 的所有 per-policy
  records。
- 添加 per-policy training step executor，对每个已附加 advantage 的
  `policy_name -> DataProto` batch 调用底层 `RayPPOTrainer` update path。
- 只更新带有当前 `DataProto` batches 的 policies。
- 在多个 trainers 之间协调 global step、metrics 和 per-policy checkpointing。

### Phase 5：Example Recipe

- 添加一个小型 blackbox multi-agent example。
- 提供 config template、reward plugin 和 debug script，其中 config template 应展示
  roles 到 policies 的映射，以及每个 policy 的 `ppo_trainer_config_name` 和
  `ppo_trainer_overrides`。
- 文档说明 example 如何将 roles 映射到 policies。
