import asyncio
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Self

from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import CreateBashSessionRequest, IsAliveResponse
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import YRDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime, RemoteRuntimeConfig

__all__ = ["YRDeployment"]

DEFAULT_SWEREX_REMOTE_LOG_PATH = "./swerex-remote.log"
DEFAULT_OPENYUANRONG_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


def _configure_env() -> None:
    """Map OPENYUANRONG_* env vars to names required by akernel-sdk before use."""
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    if not server:
        raise ValueError("OPENYUANRONG_SERVER_ADDRESS environment variable must be set")
    if not token:
        raise ValueError("OPENYUANRONG_TOKEN environment variable must be set")
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token


def _with_default_pip_index_url(env: dict[str, str] | None) -> dict[str, str]:
    merged = dict(env or {})
    merged.setdefault("PIP_INDEX_URL", DEFAULT_OPENYUANRONG_PIP_INDEX_URL)
    return merged


def _create_sandbox(config: YRDeploymentConfig) -> Any:
    """Create sandbox with port_forwardings=[port] (Port Forwarding, sandbox-api)."""
    _configure_env()
    from akernel_sdk import Sandbox

    kwargs: dict[str, Any] = {
        "cpu": config.cpu,
        "memory": config.memory,
        "cpu_limit": config.cpu_limit,
        "mem_limit": config.mem_limit,
        "idle_timeout": config.idle_timeout,
    }
    if config.image is not None:
        kwargs["image"] = config.image
    if config.env is not None:
        kwargs["env"] = dict(config.env)
    if config.name is not None:
        kwargs["name"] = config.name
    if config.cwd is not None:
        kwargs["cwd"] = config.cwd
    if config.swerex_runtime_image:
        if "mounts" in config.sandbox_kwargs:
            raise ValueError("Use either YRDeploymentConfig.swerex_runtime_image or sandbox_kwargs['mounts'], not both")
        from akernel_sdk import Mount

        kwargs["mounts"] = [
            Mount(target=config.swerex_runtime_target, image_url=config.swerex_runtime_image)
        ]
    kwargs.update(config.sandbox_kwargs)
    kwargs["env"] = _with_default_pip_index_url(kwargs.get("env"))
    kwargs["port_forwardings"] = [config.port]

    return Sandbox(**kwargs)


def _local_swerex_log_dir() -> Path:
    return Path.cwd()


def _write_local_startup_diagnostics(run_id: str, diagnostics: Any) -> Path:
    safe_run_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in run_id)
    output_dir = _local_swerex_log_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"swerex-startup-diagnostics-{safe_run_id}.log"
    output_path.write_text(
        "OpenYuanRong runtime startup diagnostics\n"
        f"exit_code={getattr(diagnostics, 'exit_code', None)}\n\n"
        "stdout:\n"
        f"{getattr(diagnostics, 'stdout', '')}\n\n"
        "stderr:\n"
        f"{getattr(diagnostics, 'stderr', '')}\n",
        encoding="utf-8",
    )
    return output_path


def _start_swerex_via_port_forwarding(
    sandbox: Any,
    *,
    command: str,
    port: int,
    internal: bool,
    log_path: str | None = None,
) -> tuple[Any, str]:
    """Port Forwarding flow from sandbox-api.md:

    1. Sandbox created with port_forwardings=[port]
    2. commands.run(server_cmd, background=True)  — start swerex on the forwarded port
    3. get_port_url(port) — gateway URL for RemoteRuntime
    """
    log_path = log_path or DEFAULT_SWEREX_REMOTE_LOG_PATH
    quoted_log_path = shlex.quote(log_path)
    command_with_logs = f"rm -f {quoted_log_path}; ( {command} ) > {quoted_log_path} 2>&1"
    handle = sandbox.commands.run(command_with_logs, background=True)
    url = sandbox.get_port_url(port, internal=internal).replace("http://", "https://")
    return handle, url


def _runtime_debug_enabled() -> bool:
    value = os.getenv("DEBUG_MODE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _debug_python_runtime(sandbox: Any) -> Any:
    command = r"""
set +e

print_python() {
    label="$1"
    py="$2"
    if [ -x "$py" ]; then
        resolved="$py"
    elif command -v "$py" >/dev/null 2>&1; then
        resolved="$(command -v "$py" 2>/dev/null)"
    else
        echo "$label=missing"
        return 0
    fi

    version="$("$resolved" --version 2>&1 || true)"
    echo "$label=$resolved ($version)"
}

print_sidecar_swerex() {
    py="/opt/swe-rex/bin/python"
    if [ ! -x "$py" ]; then
        echo "sidecar_swerex=missing-python"
        return 0
    fi

    "$py" - <<'PY' 2>&1 || true
import importlib.util

spec = importlib.util.find_spec("swerex")
if spec is None:
    print("sidecar_swerex=missing")
else:
    import swerex

    print(f"sidecar_swerex={getattr(swerex, '__version__', 'unknown')} ({spec.origin})")
PY
}

print_python "task_python" "/usr/bin/python3"
print_python "sidecar_python" "/opt/swe-rex/bin/python"
print_sidecar_swerex

if [ -x /opt/swe-rex/bin/swerex-remote ]; then
    echo "swerex_remote=/opt/swe-rex/bin/swerex-remote"
else
    echo "swerex_remote=missing"
fi
"""
    return sandbox.commands.run(command, timeout=60)


def _collect_startup_diagnostics(sandbox: Any, *, port: int, auth_token: str, log_path: str) -> Any:
    command = f"""
set +e
PORT={shlex.quote(str(port))}
AUTH_TOKEN={shlex.quote(auth_token)}
LOG_PATH={shlex.quote(log_path)}
export PORT AUTH_TOKEN LOG_PATH

echo "=== swerex startup diagnostics ==="
echo "port=$PORT"
echo "log_path=$LOG_PATH"

echo "=== swerex/python processes ==="
ps -ef | grep -E 'swerex|python' | grep -v grep || true
PID="$(pgrep -f 'swerex-remote|swerex.server' | head -1 || true)"
echo "swerex_pid=$PID"
if [ -n "$PID" ]; then
    echo "swerex_exe=$(readlink -f /proc/$PID/exe 2>/dev/null || true)"
    printf 'swerex_cmdline='
    tr '\\0' ' ' < /proc/$PID/cmdline 2>/dev/null || true
    echo
fi

echo "=== listening sockets ==="
(ss -ltnp 2>&1 || netstat -ltnp 2>&1 || true) | grep -E "(:$PORT|Local Address|LISTEN)" || true

echo "=== sandbox-local /is_alive check ==="
DEBUG_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
echo "debug_python=$DEBUG_PY"
if [ -n "$DEBUG_PY" ]; then
    "$DEBUG_PY" - <<'PY' 2>&1 || true
import os
import urllib.error
import urllib.request

port = os.environ["PORT"]
token = os.environ["AUTH_TOKEN"]
url = f"http://127.0.0.1:{{port}}/is_alive"
req = urllib.request.Request(url, headers={{"X-API-Key": token}})
try:
    with urllib.request.urlopen(req, timeout=5) as response:
        print("status=", response.status)
        print(response.read().decode(errors="replace"))
except Exception as exc:
    print(f"is_alive_error={{type(exc).__name__}}: {{exc}}")
    if isinstance(exc, urllib.error.HTTPError):
        print(exc.read().decode(errors="replace"))
PY
else
    echo "No python/python3 available for local /is_alive probe"
fi

echo "=== swerex background log ==="
if [ -f "$LOG_PATH" ]; then
    tail -200 "$LOG_PATH"
else
    echo "missing $LOG_PATH"
fi
"""
    return sandbox.commands.run(command, timeout=30)


def _kill_sandbox(sandbox: Any) -> None:
    sandbox.kill()


class YRDeployment(AbstractDeployment):
    """YR (AKernel) deployment: sandbox + Port Forwarding + swe-rex + RemoteRuntime."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = YRDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self._sandbox: Any | None = None
        self._command_handle: Any | None = None
        self._runtime_url: str | None = None
        self._port = self._config.port
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: YRDeploymentConfig, run_id: str | None = None) -> Self:
        if run_id is None:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _swerex_start_command(self, token: str) -> str:
        """Command that binds swerex.server to the port-forwarded port inside the sandbox."""
        if self._config.command:
            return self._config.command.format(token=token, port=self._port)
        rex_args = f"--host 0.0.0.0 --port {self._port} --auth-token {token}"
        prepare_cmd = os.getenv("OPENYUANRONG_ENV_PREPARE_CMD")
        if prepare_cmd:
            return (
                f"{prepare_cmd} && "
                f"({REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args})"
            )
        return f"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}"

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None or self._sandbox is None:
            raise DeploymentNotStartedError()
        loop = asyncio.get_running_loop()
        running = await loop.run_in_executor(None, self._sandbox.is_running)
        if not running:
            msg = f"YR sandbox {self._sandbox.sandbox_id} is not running"
            return IsAliveResponse(is_alive=False, message=msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        assert self._runtime is not None
        return await _wait_until_alive(
            self.is_alive,
            timeout=timeout,
            function_timeout=self._runtime._config.timeout,
        )

    async def _start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        self._stopped = False
        loop = asyncio.get_running_loop()
        token = self._get_token()
        swerex_cmd = self._swerex_start_command(token)
        swerex_remote_log_path = DEFAULT_SWEREX_REMOTE_LOG_PATH

        self.logger.info(
            f"Starting YR sandbox (port={self._port}, port_forwardings=[{self._port}], "
            f"image={self._config.image!r}, cpu={self._config.cpu}, memory={self._config.memory}), "
            f"swerex_cmd: {swerex_cmd}, remote_log_path={swerex_remote_log_path}"
        )
        self._hooks.on_custom_step("Creating YR sandbox (port forwarding)")
        t0 = time.time()

        self._sandbox = await loop.run_in_executor(None, _create_sandbox, self._config)
        elapsed_sandbox_creation = time.time() - t0
        self.logger.info(f"Sandbox {self._sandbox.sandbox_id} created in {elapsed_sandbox_creation:.2f}s")

        if _runtime_debug_enabled():
            self.logger.info("OpenYuanRong runtime debug enabled")
            debug_result = await loop.run_in_executor(None, _debug_python_runtime, self._sandbox)
            debug_stdout = getattr(debug_result, "stdout", "").strip()
            debug_stderr = getattr(debug_result, "stderr", "").strip()
            if debug_stderr:
                self.logger.info(
                    "OpenYuanRong runtime debug (exit_code={})\n{}\nstderr:\n{}",
                    getattr(debug_result, "exit_code", None),
                    debug_stdout,
                    debug_stderr,
                )
            else:
                self.logger.info(
                    "OpenYuanRong runtime debug (exit_code={})\n{}",
                    getattr(debug_result, "exit_code", None),
                    debug_stdout,
                )

        self._hooks.on_custom_step(f"Starting swerex on port {self._port} (background)")
        self._command_handle, self._runtime_url = await loop.run_in_executor(
            None,
            lambda: _start_swerex_via_port_forwarding(
                self._sandbox,
                command=swerex_cmd,
                port=self._port,
                internal=self._config.internal,
                log_path=swerex_remote_log_path,
            ),
        )

        self.logger.info(f"Port forwarding URL: {self._runtime_url}")
        self._hooks.on_custom_step("Connecting RemoteRuntime via port forwarding")

        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(
                host=self._runtime_url,
                timeout=self._config.timeout,
                auth_token=token,
                proxy=self._config.proxy,
                ssl_verify=self._config.ssl_verify,
            ),
            run_id=self.run_id,
        )

        remaining_startup_timeout = max(0, self._config.startup_timeout - elapsed_sandbox_creation)
        t1 = time.time()
        try:
            await self._wait_until_alive(timeout=remaining_startup_timeout)
            await self.runtime.create_session(CreateBashSessionRequest(startup_timeout=60))
        except Exception:
            diagnostics = await loop.run_in_executor(
                None,
                lambda: _collect_startup_diagnostics(
                    self._sandbox,
                    port=self._port,
                    auth_token=token,
                    log_path=swerex_remote_log_path,
                ),
            )
            diagnostics_path = await loop.run_in_executor(
                None,
                lambda: _write_local_startup_diagnostics(self.run_id, diagnostics),
            )
            self.logger.error(
                "OpenYuanRong runtime startup diagnostics written to {}\n"
                "OpenYuanRong runtime startup diagnostics (exit_code={})\nstdout:\n{}\nstderr:\n{}",
                diagnostics_path,
                getattr(diagnostics, "exit_code", None),
                getattr(diagnostics, "stdout", ""),
                getattr(diagnostics, "stderr", ""),
            )
            raise
        self.logger.info(f"Runtime started in {time.time() - t1:.2f}s")

    async def start(self, max_retries: int = 5) -> None:
        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                await self._start()
                return
            except Exception as exc:
                last_error = exc
                self.logger.critical(f"Failed to create YR sandbox: {exc}")
                await self.stop()
                if retry < max_retries - 1:
                    sleep_time = min(30, 2**retry)
                    self.logger.info(f"Retrying YR deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create YR sandbox after {max_retries} retries") from last_error

    async def stop(self) -> None:
        if self._stopped:
            return

        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as e:
                self.logger.error(f"Failed to close YR runtime: {e}")
            self._runtime = None

        if self._command_handle is not None:
            try:
                self._command_handle.kill()
            except Exception as e:
                self.logger.debug(f"Failed to kill swerex background process: {e}")
            self._command_handle = None

        if self._sandbox is not None:
            loop = asyncio.get_running_loop()
            sandbox_id = self._sandbox.sandbox_id
            try:
                if self._sandbox.is_running():
                    await loop.run_in_executor(None, _kill_sandbox, self._sandbox)
                    self.logger.info(f"Sandbox {sandbox_id} killed")
            except Exception as e:
                self.logger.error(f"Failed to kill sandbox {sandbox_id}: {e}")
            self._sandbox = None

        self._runtime_url = None
        self._stopped = True

    @property
    def runtime(self) -> RemoteRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            raise DeploymentNotStartedError()
        return self._sandbox

    @property
    def port_forward_url(self) -> str:
        """Gateway URL for the forwarded swerex port (from get_port_url)."""
        if self._runtime_url is None:
            raise DeploymentNotStartedError()
        return self._runtime_url

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
