"""Agent framework, gateway, and trainer orchestration packages."""

__all__ = ["MultiAgentsPPOTrainer"]


def __getattr__(name):
    if name == "MultiAgentsPPOTrainer":
        from .multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        return MultiAgentsPPOTrainer
    raise AttributeError(name)

