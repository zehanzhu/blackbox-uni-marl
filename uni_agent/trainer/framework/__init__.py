from .types import SessionHandle, Trajectory

__all__ = [
    "AgentFramework",
    "AnthropicCompatibleAgentFramework",
    "MultiAgentFramework",
    "OpenAICompatibleAgentFramework",
    "SessionHandle",
    "Trajectory",
]


def __getattr__(name):
    if name in {
        "AgentFramework",
        "AnthropicCompatibleAgentFramework",
        "MultiAgentFramework",
        "OpenAICompatibleAgentFramework",
    }:
        from .framework import (
            AgentFramework,
            AnthropicCompatibleAgentFramework,
            MultiAgentFramework,
            OpenAICompatibleAgentFramework,
        )

        return {
            "AgentFramework": AgentFramework,
            "AnthropicCompatibleAgentFramework": AnthropicCompatibleAgentFramework,
            "MultiAgentFramework": MultiAgentFramework,
            "OpenAICompatibleAgentFramework": OpenAICompatibleAgentFramework,
        }[name]
    raise AttributeError(name)
