# Multi-Agent Blackbox Worker Setup Hook Design

## Context

`examples/multi_agent_blackbox` starts multiple verl v1 PPO trainers in one
Ray job. The example-owned runtime patch gives each policy unique placement
group, reward worker, and vLLM server names without modifying the shared verl
checkout.

The current patch is installed inside `MultiAgentsTaskRunner.run()`. That
correctly patches the process that creates vLLM server actors, but the
`ActorRolloutRefWorker` processes are separate Python interpreters. Their
`ServerAdapter` can therefore retain verl's original lookup prefix and request
`vllm_server_<replica>_<node>` while the creator registered
`vllm_<policy>_server_<replica>_<node>`.

## Decision

Register Ray's job-level `worker_process_setup_hook` directly in the example
configuration:

```yaml
ray_kwargs:
  ray_init:
    num_cpus: null
    runtime_env:
      worker_process_setup_hook: examples.multi_agent_blackbox.verl_patch.apply_worker_patch
```

Ray executes the dotted callable once when each Python worker process starts,
before scheduling tasks or actors in that process. The hook installs only the
`ServerAdapter` lookup-side patch. The existing `apply_patch()` call in the
TaskRunner remains responsible for placement groups, reward workers, and the
vLLM creator side.

No file under `D:\RFC\verl` is modified.

## Runtime Flow

```text
Training driver
  -> ray.init(worker_process_setup_hook=apply_worker_patch)
     -> MultiAgentsTaskRunner process
        -> apply_worker_patch() before actor code
        -> existing apply_patch() in run()
           -> placement-group uniqueness
           -> reward-worker uniqueness
           -> vLLMReplica creator prefix
     -> ActorRolloutRefWorker processes
        -> apply_worker_patch() before actor code
           -> ServerAdapter lookup prefix
     -> gateway/reward/other Python workers
        -> apply_worker_patch() once; patched class is unused there
```

For a rollout config carrying `custom.policy_name=policy_1`, both sides produce
the same prefix and actor name:

```text
creator prefix: policy_1_vllm_
lookup prefix:  policy_1_vllm_
actor name:     policy_1_vllm_server_0_0
```

## Patch Structure

`verl_patch.py` will expose two idempotent entry points:

```python
def apply_worker_patch() -> None:
    """Patch only ServerAdapter's policy-aware actor lookup prefix."""


def apply_patch() -> None:
    """Install TaskRunner-side resource, reward, and creator patches."""
```

The state is separated:

```python
_PATCHED = False
_WORKER_PATCHED = False
```

`apply_patch()` also invokes `apply_worker_patch()` when the vLLM rollout
module is available. This preserves direct-use and unit-test behavior while
remaining a no-op in a TaskRunner process where the Ray setup hook already ran.

Imports for resource-pool and reward-worker patches move out of module scope
and into their owning code paths. This avoids loading unrelated trainer
components merely to resolve the dotted worker hook. Importing `ServerAdapter`
still loads Torch and a small set of verl rollout modules; this is a one-time
per-process startup cost.

## Lifecycle And Failure Behavior

`apply_worker_patch()` is transactional:

1. Import `ServerAdapter`.
2. Capture its original `_get_server_name_prefix`.
3. Replace the method.
4. Mark `_WORKER_PATCHED` only after successful replacement.

An import or installation error propagates out of the setup hook. Ray then
reports a runtime-environment setup failure instead of allowing training to
continue until a less diagnostic actor lookup error occurs.

Repeated calls in one process are no-ops. `restore()` restores both the full
TaskRunner patch and the worker patch for test isolation.

The first version intentionally does not handle a Python driver in which Ray
was initialized before `run_multi_agents_ppo()`. The supported launch script
starts a fresh Python process, so its first `ray.init()` carries the hook.

Ray permits one worker setup hook per runtime environment. A future second hook
must be composed into an example-owned aggregate callable rather than replacing
this hook silently.

## Gateway Impact

The hook applies to every Python worker in the Ray job. The current Gateway
pool is fixed by `gateway_count: 4` and is created once before `trainer.fit()`.
Training creates dynamic sessions inside those four actors; it does not create
additional Gateway actors.

Gateway actors do not instantiate `ServerAdapter`, so the patch has no behavior
there after startup. The only marginal cost is the one-time module import. No
GPU allocation, Ray actor creation, network call, or per-request work is added
by the hook itself.

## Testing

### Configuration Contract

Assert that the example YAML contains the exact dotted hook path and that the
existing runtime-env merge passes it to `ray.init()` alongside env vars.

### Patch Unit Tests

Use fake verl classes to verify:

- policy-aware and fallback prefixes;
- creator/lookup equality;
- worker-patch idempotence;
- full-patch composition with the worker patch;
- restoration of original methods;
- no partial installed state after an import failure.

### Ray Integration

Update `verify_vllm_servers.py` to initialize Ray with the same hook. Launch a
small remote probe that confirms the worker-local `ServerAdapter` produces
`policy_1_vllm_`, then initialize both real policy trainers.

The GPU smoke succeeds only when:

- both policy trainers complete `init()` and initial weight synchronization;
- placement-group and reward actor names do not collide;
- both policies expose distinct vLLM server actors;
- no `Failed to look up actor` error occurs.

## Deployment And Rollback

All Ray nodes must be able to import the repository through a shared checkout,
editable installation, or equivalent `PYTHONPATH`. The existing Ray cluster
does not need to restart; a new training driver is required so a new Ray job is
created with the hook.

Roll back by removing the YAML hook and reverting the worker-patch additions,
then start a new training driver. The independent Ray cluster startup script is
unchanged.
