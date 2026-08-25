__all__ = [
    "GatewayActor",
    "GatewayManager",
    "GatewayServingRuntime",
    "PolicyRoutingLLMClient",
]


def __getattr__(name):
    if name == "GatewayActor":
        from .gateway import GatewayActor

        return GatewayActor
    if name == "GatewayManager":
        from .manager import GatewayManager

        return GatewayManager
    if name in {"GatewayServingRuntime", "PolicyRoutingLLMClient"}:
        from .runtime import GatewayServingRuntime, PolicyRoutingLLMClient

        return {
            "GatewayServingRuntime": GatewayServingRuntime,
            "PolicyRoutingLLMClient": PolicyRoutingLLMClient,
        }[name]
    raise AttributeError(name)
