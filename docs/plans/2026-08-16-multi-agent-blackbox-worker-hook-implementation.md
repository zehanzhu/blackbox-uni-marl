# Multi-Agent Blackbox Worker Setup Hook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure every Ray rollout worker uses the same policy-qualified vLLM actor name as the creator without modifying verl.

**Architecture:** Register an example-owned dotted `worker_process_setup_hook` in the existing Ray runtime environment. Split the current runtime patch into a narrow worker-side `ServerAdapter` patch and the existing TaskRunner-side resource/reward/creator patch, with independent idempotence and restoration.

**Tech Stack:** Python 3.10+, Ray 2.55.1 runtime environments, Hydra/OmegaConf YAML, pytest, verl v1 PPO trainer and vLLM rollout adapter.

---

### Task 1: Declare The Worker Hook In Example Configuration

**Files:**
- Modify: `tests/test_multi_agent_blackbox_example.py:184`
- Modify: `examples/multi_agent_blackbox/config/multi_agent_blackbox.yaml:70`
- Modify: `tests/uni_agent/trainer/test_main_multi_agents_ppo.py:132`

**Step 1: Write the failing YAML contract assertion**

Update `test_multi_agent_blackbox_yaml_has_v1_compatible_transfer_queue_and_ray_defaults` so the expected Ray config is:

```python
assert cfg["ray_kwargs"] == {
    "ray_init": {
        "num_cpus": None,
        "runtime_env": {
            "worker_process_setup_hook": (
                "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
            ),
        },
    },
    "timeline_json_file": None,
}
```

**Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest tests/test_multi_agent_blackbox_example.py::test_multi_agent_blackbox_yaml_has_v1_compatible_transfer_queue_and_ray_defaults -q
```

Expected: FAIL because `runtime_env.worker_process_setup_hook` is absent.

**Step 3: Add the hook to the example YAML**

Change the Ray block to:

```yaml
ray_kwargs:
  ray_init:
    num_cpus: null
    runtime_env:
      worker_process_setup_hook: examples.multi_agent_blackbox.verl_patch.apply_worker_patch
  timeline_json_file: null
```

**Step 4: Verify the existing runtime-env merge preserves the hook**

In `test_run_multi_agents_ppo_initializes_ray_and_runs_remote_task`, add the
hook to the test input:

```python
runtime_env=ConfigNode(
    env_vars=ConfigNode(EXISTING_ENV="1"),
    worker_process_setup_hook=(
        "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
    ),
),
```

Add the same key to the expected `fake_ray.init_calls[0]["runtime_env"]`.
No production entry-point change should be necessary.

**Step 5: Run both focused tests**

Run:

```bash
python -m pytest \
  tests/test_multi_agent_blackbox_example.py::test_multi_agent_blackbox_yaml_has_v1_compatible_transfer_queue_and_ray_defaults \
  tests/uni_agent/trainer/test_main_multi_agents_ppo.py::test_run_multi_agents_ppo_initializes_ray_and_runs_remote_task \
  -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add examples/multi_agent_blackbox/config/multi_agent_blackbox.yaml \
  tests/test_multi_agent_blackbox_example.py \
  tests/uni_agent/trainer/test_main_multi_agents_ppo.py
git commit -m "config: register blackbox Ray worker hook"
```

### Task 2: Add An Independently Idempotent Worker Patch

**Files:**
- Create: `tests/test_multi_agent_blackbox_verl_patch.py`
- Modify: `examples/multi_agent_blackbox/verl_patch.py:40`

**Step 1: Add a fake-module test harness**

Load `verl_patch.py` with fake target modules in `sys.modules`. The fake lookup
class should have an observable original method:

```python
class FakeServerAdapter:
    def _get_server_name_prefix(self):
        return "original_"


class AttrDict(dict):
    __getattr__ = dict.__getitem__
```

Install it under:

```python
"verl.workers.rollout.vllm_rollout.vllm_rollout"
```

Ensure each test imports a fresh patch module or calls `restore()` in teardown.

**Step 2: Write the failing worker-prefix test**

```python
def test_apply_worker_patch_uses_policy_qualified_prefix(patch_module):
    patch_module.apply_worker_patch()
    adapter = FakeServerAdapter()
    adapter.config = AttrDict(custom={"policy_name": "policy_1"})

    assert adapter._get_server_name_prefix() == "policy_1_vllm_"
```

Also assert fallback behavior with `custom={}` returns `vllm_`.

**Step 3: Write failing idempotence and restore tests**

```python
def test_worker_patch_is_idempotent_and_restorable(patch_module):
    original = FakeServerAdapter._get_server_name_prefix

    patch_module.apply_worker_patch()
    installed = FakeServerAdapter._get_server_name_prefix
    patch_module.apply_worker_patch()

    assert FakeServerAdapter._get_server_name_prefix is installed

    patch_module.restore()
    assert FakeServerAdapter._get_server_name_prefix is original
```

Add a test that a failed target import leaves `_WORKER_PATCHED` false.

**Step 4: Run the new tests and verify they fail**

Run:

```bash
python -m pytest tests/test_multi_agent_blackbox_verl_patch.py -q
```

Expected: FAIL because `apply_worker_patch` and `_WORKER_PATCHED` do not exist.

**Step 5: Refactor module-level imports and state**

Remove eager Ray/verl resource-pool imports from module scope. Initialize
component originals lazily:

```python
_ORIG_INIT = None
_ORIG_CREATE = None
_ORIG_INIT_RESOURCE_POOL_MGR = None
_ORIG_INIT_REWARD_LOOP_WORKERS = None
_ORIG_VLLM_REPLICA_NAME_PREFIX = None
_ORIG_SERVER_ADAPTER_NAME_PREFIX = None
_PATCHED = False
_WORKER_PATCHED = False
```

Import `ray` inside `_patched_init_reward_loop_workers`. Import
`RayResourcePool` and `ResourcePoolManager` inside `apply_patch()` and
`restore()`.

**Step 6: Implement the narrow worker hook**

```python
def apply_worker_patch() -> None:
    """Install the policy-aware vLLM lookup prefix in this process."""
    global _WORKER_PATCHED, _ORIG_SERVER_ADAPTER_NAME_PREFIX
    if _WORKER_PATCHED:
        return

    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    original = ServerAdapter._get_server_name_prefix
    ServerAdapter._get_server_name_prefix = _patched_server_adapter_get_server_name_prefix
    _ORIG_SERVER_ADAPTER_NAME_PREFIX = original
    _WORKER_PATCHED = True
    logger.info("multi_agent_blackbox.verl_patch: worker lookup patch enabled")
```

Assign the original before setting `_WORKER_PATCHED`; never swallow hook import
or assignment errors.

**Step 7: Compose it with the full patch**

Keep creator installation in `apply_patch()` and replace its direct adapter
assignment with:

```python
if ServerAdapter is not None:
    apply_worker_patch()
```

The full patch may retain its existing warning behavior for optional vLLM
imports. A direct invocation still installs both creator and lookup methods
when vLLM is available.

**Step 8: Restore worker state independently**

Extend `restore()`:

```python
global _WORKER_PATCHED, _ORIG_SERVER_ADAPTER_NAME_PREFIX
if _WORKER_PATCHED and _ORIG_SERVER_ADAPTER_NAME_PREFIX is not None:
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    ServerAdapter._get_server_name_prefix = _ORIG_SERVER_ADAPTER_NAME_PREFIX
    _ORIG_SERVER_ADAPTER_NAME_PREFIX = None
    _WORKER_PATCHED = False
```

Restore full-patch components as today and reset `_PATCHED` separately.

**Step 9: Run patch tests**

Run:

```bash
python -m pytest tests/test_multi_agent_blackbox_verl_patch.py -q
```

Expected: PASS.

**Step 10: Commit**

```bash
git add examples/multi_agent_blackbox/verl_patch.py \
  tests/test_multi_agent_blackbox_verl_patch.py
git commit -m "fix: install vLLM lookup patch in Ray workers"
```

### Task 3: Prove Creator And Lookup Names Stay Equal

**Files:**
- Modify: `tests/test_multi_agent_blackbox_verl_patch.py`

**Step 1: Add the failing equality test**

Provide fake `vLLMReplica` and `ServerAdapter` objects with the same rollout
config:

```python
def test_creator_and_lookup_prefixes_match_for_each_policy(patch_module):
    patch_module.apply_patch()

    for policy_name in ("policy_1", "policy_2"):
        config = AttrDict(custom={"policy_name": policy_name})
        replica = FakeVLLMReplica()
        replica.config = config
        adapter = FakeServerAdapter()
        adapter.config = config

        assert replica._get_server_name_prefix() == adapter._get_server_name_prefix()
        assert replica._get_server_name_prefix() == f"{policy_name}_vllm_"
```

**Step 2: Run the test**

Run:

```bash
python -m pytest \
  tests/test_multi_agent_blackbox_verl_patch.py::test_creator_and_lookup_prefixes_match_for_each_policy \
  -q
```

Expected: PASS after Task 2; if it fails, correct the shared policy-label helper
rather than adding separate naming logic.

**Step 3: Run the whole patch test file**

```bash
python -m pytest tests/test_multi_agent_blackbox_verl_patch.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_multi_agent_blackbox_verl_patch.py
git commit -m "test: lock vLLM creator lookup name parity"
```

### Task 4: Add A Real Ray Worker Probe

**Files:**
- Modify: `examples/multi_agent_blackbox/scripts/verify_vllm_servers.py:57`

**Step 1: Register the production hook in the verifier**

Change `_connect()` to initialize Ray with:

```python
ray.init(
    address="auto",
    log_to_driver=False,
    runtime_env={
        "worker_process_setup_hook": (
            "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
        )
    },
)
```

**Step 2: Add a remote lookup-prefix probe**

After `_connect()`, define or invoke a remote function that runs in a separate
Ray worker:

```python
@ray.remote
def probe_worker_lookup_prefix():
    from types import SimpleNamespace
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    adapter = object.__new__(ServerAdapter)
    adapter.config = SimpleNamespace(custom={"policy_name": "policy_1"})
    return {
        "pid": os.getpid(),
        "prefix": adapter._get_server_name_prefix(),
    }
```

Accept dict-like and attribute-style rollout custom config in the helper, or
construct the same dataclass/config shape used by the installed verl version.

**Step 3: Fail before allocating GPUs when the probe is wrong**

```python
probe = ray.get(probe_worker_lookup_prefix.remote())
if probe["prefix"] != "policy_1_vllm_":
    raise SystemExit(f"Ray worker lookup patch missing: {probe}")
print(f"worker patch probe: {probe}")
```

Run this before constructing `MultiAgentsPPOTrainer` so a hook regression fails
without launching policy models.

**Step 4: Keep the existing real trainer initialization check**

The existing `trainer.init()` remains the end-to-end assertion. It exercises
initial `PPOTrainerSync.on_init_end()` weight synchronization, where an
unpatched lookup previously fails.

**Step 5: Syntax-check the verifier locally**

Run:

```bash
python -m py_compile examples/multi_agent_blackbox/scripts/verify_vllm_servers.py
```

Expected: PASS. Do not claim the Ray/GPU path passed on the Windows workspace.

**Step 6: Commit**

```bash
git add examples/multi_agent_blackbox/scripts/verify_vllm_servers.py
git commit -m "test: verify blackbox patch in Ray worker"
```

### Task 5: Document Operations And Run Focused Regression Tests

**Files:**
- Modify: `examples/multi_agent_blackbox/README.md:24`

**Step 1: Document the two-process patch model**

Add a short section explaining:

```text
TaskRunner apply_patch(): resource pools, reward actors, vLLM creator
Ray worker setup hook: ServerAdapter lookup
```

Document that all nodes must import the repository, Gateway count remains
fixed, the hook is one-time per Python worker, and a fresh training driver is
required after changing the hook.

**Step 2: Document verification commands**

Include:

```bash
bash examples/multi_agent_blackbox/scripts/run_verify_vllm_servers.sh
bash examples/multi_agent_blackbox/scripts/run_train.sh
```

State that the verifier should report `policy_1_vllm_` from a remote worker
before GPU initialization.

**Step 3: Run focused unit tests**

```bash
python -m pytest \
  tests/test_multi_agent_blackbox_example.py \
  tests/test_multi_agent_blackbox_verl_patch.py \
  tests/uni_agent/trainer/test_main_multi_agents_ppo.py \
  -q
```

Expected: PASS.

**Step 4: Run the broader trainer regression set**

```bash
python -m pytest \
  tests/uni_agent/trainer/test_multi_agents_ppo_trainer.py \
  tests/uni_agent/trainer/test_multi_agents_v1_smoke.py \
  -q
```

Expected: PASS, with only pre-existing environment-dependent skips.

**Step 5: Run the target Linux GPU smoke**

On the Ray 2.55.1 cluster with both policy model paths and datasets configured:

```bash
bash examples/multi_agent_blackbox/scripts/run_verify_vllm_servers.sh
```

Expected:

```text
worker patch probe: ... 'prefix': 'policy_1_vllm_'
OK: two policies' vLLM servers started
```

There must be no placement-group collision, reward actor collision, or
`Failed to look up actor` error.

**Step 6: Commit**

```bash
git add examples/multi_agent_blackbox/README.md
git commit -m "docs: explain blackbox Ray worker patch"
```

### Task 6: Final Diff And No-VERL Verification

**Files:**
- Verify only; no planned edits.

**Step 1: Confirm no sibling verl file changed**

```bash
git -C D:/RFC/verl status --short
```

Expected: no changes caused by this implementation. Do not alter or clean
pre-existing sibling-repository changes.

**Step 2: Review the Uni-MARL diff**

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only the files listed in Tasks 1-5, plus any pre-existing unrelated
workspace files left untouched; `git diff --check` exits zero.

**Step 3: Re-run the focused suite**

```bash
python -m pytest \
  tests/test_multi_agent_blackbox_example.py \
  tests/test_multi_agent_blackbox_verl_patch.py \
  tests/uni_agent/trainer/test_main_multi_agents_ppo.py \
  -q
```

Expected: PASS.

**Step 4: Record external verification status**

If the Linux GPU smoke has not run, state that explicitly in the completion
report. Do not infer real Ray cross-process success from unit tests alone.
