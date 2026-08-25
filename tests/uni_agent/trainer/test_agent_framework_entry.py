import asyncio
import sys
import types
from types import SimpleNamespace


def _install_dependency_stubs():
    if "omegaconf" not in sys.modules:
        omegaconf_mod = types.ModuleType("omegaconf")

        class OmegaConf:
            @staticmethod
            def select(config, key, default=None):
                return default

        omegaconf_mod.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf_mod

    if "ray" not in sys.modules:
        ray_stub = types.ModuleType("ray")
        ray_stub.actor = SimpleNamespace(ActorHandle=object)
        ray_stub.remote = lambda cls: cls
        ray_stub.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        sys.modules["ray"] = ray_stub

    if "verl.utils.config" not in sys.modules:
        for name in [
            "verl",
            "verl.tools",
            "verl.utils",
            "verl.workers",
            "verl.workers.config",
            "verl.workers.rollout",
        ]:
            sys.modules.setdefault(name, types.ModuleType(name))
        config_mod = types.ModuleType("verl.utils.config")
        config_mod.omega_conf_to_dataclass = lambda config: config
        import_utils_mod = types.ModuleType("verl.utils.import_utils")
        import_utils_mod.load_class_from_fqn = lambda fqn: FakeFramework
        ray_utils_mod = types.ModuleType("verl.utils.ray_utils")
        ray_utils_mod.auto_await = lambda func: func
        tool_registry_mod = types.ModuleType("verl.tools.tool_registry")
        tool_registry_mod.initialize_tools_from_config = lambda path: []
        transferqueue_utils_mod = types.ModuleType("verl.utils.transferqueue_utils")
        transferqueue_utils_mod.tq = SimpleNamespace()
        tensordict_utils_mod = types.ModuleType("verl.utils.tensordict_utils")
        tensordict_utils_mod.get = lambda tensor_dict, key: tensor_dict[key]
        model_utils_mod = types.ModuleType("verl.utils.model")
        model_utils_mod.compute_position_id_with_mask = lambda attention_mask: attention_mask
        model_mod = types.ModuleType("verl.workers.config.model")
        model_mod.HFModelConfig = object
        llm_server_mod = types.ModuleType("verl.workers.rollout.llm_server")
        llm_server_mod.LLMServerClient = object
        sys.modules["verl.utils.config"] = config_mod
        sys.modules["verl.utils.import_utils"] = import_utils_mod
        sys.modules["verl.utils.ray_utils"] = ray_utils_mod
        sys.modules["verl.tools.tool_registry"] = tool_registry_mod
        sys.modules["verl.utils.transferqueue_utils"] = transferqueue_utils_mod
        sys.modules["verl.utils.tensordict_utils"] = tensordict_utils_mod
        sys.modules["verl.utils.model"] = model_utils_mod
        sys.modules["verl.workers.config.model"] = model_mod
        sys.modules["verl.workers.rollout.llm_server"] = llm_server_mod


class FakeFramework:
    from_config_calls = []

    @classmethod
    async def from_config(cls, **kwargs):
        cls.from_config_calls.append(kwargs)
        return SimpleNamespace(kind="framework", kwargs=kwargs)


class FakeGatewayServingRuntime:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeGatewayServingRuntime.instances.append(self)


class TestAgentFrameworkEntry:
    def test_build_agent_framework_uses_supplied_gateway_actor_kwargs(self, monkeypatch):
        _install_dependency_stubs()

        from uni_agent.trainer.framework import entry

        FakeFramework.from_config_calls.clear()
        FakeGatewayServingRuntime.instances.clear()
        monkeypatch.setattr(entry, "GatewayServingRuntime", FakeGatewayServingRuntime)
        monkeypatch.setattr(entry, "load_class_from_fqn", lambda fqn: FakeFramework)
        monkeypatch.setattr(
            entry.OmegaConf,
            "select",
            lambda config, key, default=None: {"gateway_count": 0}
            if key == "actor_rollout_ref.rollout.custom.agent_framework"
            else default,
        )
        monkeypatch.setattr(
            entry,
            "omega_conf_to_dataclass",
            lambda config: (_ for _ in ()).throw(AssertionError("should not load model config")),
        )

        gateway_actor_kwargs = {
            "processor": "processor:default",
            "policy_tokenizers": {"policy_1": "tokenizer:policy_1"},
            "policy_processors": {"policy_1": "processor:policy_1"},
        }

        framework = asyncio.run(
            entry.build_agent_framework(
                config=SimpleNamespace(),
                llm_client="llm-client",
                replay_buffer="replay-buffer",
                reward_loop_worker_handles=["reward"],
                gateway_actor_kwargs=gateway_actor_kwargs,
            )
        )

        assert framework.kind == "framework"
        assert FakeGatewayServingRuntime.instances[0].kwargs["gateway_actor_kwargs"] == gateway_actor_kwargs
        assert FakeFramework.from_config_calls[0]["processor"] == "processor:default"
        assert FakeFramework.from_config_calls[0]["policy_processors"] == {
            "policy_1": "processor:policy_1",
        }

    def test_adapter_forwards_gateway_actor_kwargs(self):
        _install_dependency_stubs()

        from uni_agent.trainer.framework import entry

        captured = {}

        async def fake_build_agent_framework(**kwargs):
            captured.update(kwargs)
            return "framework"

        original_build = entry.build_agent_framework
        entry.build_agent_framework = fake_build_agent_framework
        try:
            adapter = asyncio.run(
                entry.AgentFrameworkRolloutAdapter.create(
                    config=SimpleNamespace(),
                    llm_client="llm-client",
                    replay_buffer="replay-buffer",
                    reward_loop_worker_handles=["reward"],
                    gateway_actor_kwargs={"policy_tokenizers": {"policy_1": "tokenizer"}},
                )
            )
        finally:
            entry.build_agent_framework = original_build

        assert adapter.framework == "framework"
        assert captured["gateway_actor_kwargs"] == {"policy_tokenizers": {"policy_1": "tokenizer"}}
