#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"
DEFAULT_LISTEN_HOST = "0.0.0.0"
DEFAULT_LISTEN_PORT = 8003
ACK_PAYLOAD = "ACK"
TOKEN_DETAILS_MIDDLEWARE = "server.TokenDetailsDefaultsMiddleware"
TOKEN_DETAILS_ENDPOINTS = {
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/chat/completions/batch",
}


class TokenDetailsDefaultsMiddleware:
    """Inject token IDs and output logprobs into vLLM generation requests."""

    def __init__(self, app: object) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, object],
        receive: object,
        send: object,
    ) -> None:
        path = str(scope.get("path", "")).rstrip("/")
        headers = scope.get("headers", [])
        content_type = ""
        content_encoding = ""
        if isinstance(headers, list):
            for key, value in headers:
                if key.lower() == b"content-type":
                    content_type = value.decode("latin-1")
                elif key.lower() == b"content-encoding":
                    content_encoding = value.decode("latin-1")

        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or path not in TOKEN_DETAILS_ENDPOINTS
            or "application/json" not in content_type.lower()
            or content_encoding
        ):
            await self.app(scope, receive, send)
            return

        body_parts: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, _single_message_receiver(message), send)
                return
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        original_body = b"".join(body_parts)
        try:
            payload = json.loads(original_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self.app(
                scope,
                _single_message_receiver(
                    {"type": "http.request", "body": original_body}
                ),
                send,
            )
            return

        if not isinstance(payload, dict):
            await self.app(
                scope,
                _single_message_receiver(
                    {"type": "http.request", "body": original_body}
                ),
                send,
            )
            return

        payload.setdefault("return_token_ids", True)
        if path == "/v1/completions":
            payload.setdefault("logprobs", 1)
        else:
            payload.setdefault("logprobs", True)
            if payload.get("logprobs"):
                payload.setdefault("top_logprobs", 1)

        modified_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        modified_scope = dict(scope)
        modified_headers = [
            (key, value)
            for key, value in headers
            if key.lower() != b"content-length"
        ]
        modified_headers.append((b"content-length", str(len(modified_body)).encode()))
        modified_scope["headers"] = modified_headers

        await self.app(
            modified_scope,
            _single_message_receiver(
                {"type": "http.request", "body": modified_body}
            ),
            send,
        )


def _single_message_receiver(message: dict[str, object]) -> object:
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return message

    return receive


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    if not path.exists():
        raise FileNotFoundError(f"missing env file: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        values[key] = value

    return values


def default_backend_script() -> str:
    env_backend = os.environ.get("VLLM_BACKEND_SCRIPT")
    if env_backend:
        return env_backend
    if (SCRIPT_DIR / "vllm.py").exists():
        return "vllm.py"
    return "vllm.sh"


def resolve_backend_script(script: str) -> Path:
    path = Path(script)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"backend script does not exist: {path}")

    return path


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the vLLM backend as a subprocess, ACK the CPU machine, "
            "and stop the backend when an ACK is received."
        )
    )
    parser.add_argument(
        "--backend-script",
        default=default_backend_script(),
        help="Backend script to run with bash. Default: vllm.py if present, otherwise vllm.sh.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or local path to pass to the vLLM backend.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="vLLM backend port. If omitted, VLLM_PORT is required in ../.env.",
    )
    parser.add_argument(
        "--listen-host",
        default=DEFAULT_LISTEN_HOST,
        help=f"ACK listener host. Default: {DEFAULT_LISTEN_HOST}",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=DEFAULT_LISTEN_PORT,
        help=f"ACK listener port. Default: {DEFAULT_LISTEN_PORT}",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the outbound ACK request. Default: 5.0",
    )
    parser.add_argument(
        "--ack-retries",
        type=int,
        default=3,
        help="Outbound ACK attempts before continuing. Default: 3",
    )
    parser.add_argument(
        "--stop-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait after SIGINT before escalating to SIGTERM. Default: 30.0",
    )

    args, backend_args = parser.parse_known_args(argv)
    if backend_args and backend_args[0] == "--":
        backend_args = backend_args[1:]
    return args, backend_args


def validate_port(port: int, name: str) -> int:
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be in [1, 65535], got {port}")
    return port


def resolve_vllm_port(
    explicit_port: int | None, env_values: dict[str, str]
) -> int | None:
    if explicit_port is not None:
        return validate_port(explicit_port, "--vllm-port")

    env_port = env_values.get("VLLM_PORT")
    if not env_port:
        raise RuntimeError(f"VLLM_PORT is not set in {ENV_PATH}")

    try:
        return validate_port(int(env_port), "VLLM_PORT")
    except ValueError as exc:
        raise ValueError(f"invalid VLLM_PORT in {ENV_PATH}: {env_port!r}") from exc


def make_backend_args(
    passthrough_args: list[str], model: str | None, vllm_port: int | None
) -> list[str]:
    backend_args: list[str] = []

    if model:
        backend_args.extend(["--model", model])
    if vllm_port is not None:
        backend_args.extend(["--port", str(vllm_port)])

    backend_args.extend(["--", "--middleware", TOKEN_DETAILS_MIDDLEWARE])
    backend_args.extend(passthrough_args)
    return backend_args


def send_ack(cpu_machine_url: str, timeout: float, retries: int) -> bool:
    retries = max(retries, 1)
    payload = ACK_PAYLOAD.encode("utf-8")

    for attempt in range(1, retries + 1):
        request = Request(
            cpu_machine_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                response.read(256)
                print(f"Sent ACK to CPU machine, status={response.status}", flush=True)
                return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(
                f"ACK send attempt {attempt}/{retries} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(1.0)

    return False


def wait_for_vllm_health(
    process: subprocess.Popen[bytes],
    vllm_port: int,
    request_timeout: float = 2.0,
    poll_interval: float = 1.0,
) -> None:
    health_url = f"http://127.0.0.1:{vllm_port}/health"
    last_error: Exception | None = None
    print(f"Waiting for vLLM health check: {health_url}", flush=True)

    while process.poll() is None:
        request = Request(health_url, method="GET")
        try:
            with urlopen(request, timeout=request_timeout) as response:
                response.read(256)
                if 200 <= response.status < 300:
                    print(
                        f"vLLM health check passed, status={response.status}",
                        flush=True,
                    )
                    return
                last_error = RuntimeError(
                    f"health endpoint returned status {response.status}"
                )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc

        time.sleep(poll_interval)

    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(
        f"vLLM backend exited with code {process.returncode} before becoming healthy"
        f"{detail}"
    )


def is_ack_request(path: str, body: bytes) -> bool:
    parsed = urlparse(path)
    body_text = body.decode("utf-8", errors="replace").strip()
    path_text = parsed.path.strip("/")
    query = parse_qs(parsed.query)

    if body_text.upper() == ACK_PAYLOAD:
        return True
    if path_text.upper() == ACK_PAYLOAD:
        return True

    return any(
        value.upper() == ACK_PAYLOAD
        for values in query.values()
        for value in values
    )


def signal_process_group(
    process: subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    if process.poll() is not None:
        return

    if hasattr(os, "killpg"):
        os.killpg(process.pid, sig)
    else:
        process.send_signal(sig)


def stop_backend(
    process: subprocess.Popen[bytes], stop_timeout: float, reason: str
) -> None:
    if process.poll() is not None:
        print(f"Backend already exited with code {process.returncode}", flush=True)
        return

    print(f"{reason}; sending SIGINT to backend", flush=True)
    signal_process_group(process, signal.SIGINT)

    try:
        process.wait(timeout=stop_timeout)
        print(f"Backend exited with code {process.returncode}", flush=True)
        return
    except subprocess.TimeoutExpired:
        print("Backend did not exit after SIGINT; sending SIGTERM", flush=True)

    signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
        print(f"Backend exited with code {process.returncode}", flush=True)
        return
    except subprocess.TimeoutExpired:
        print("Backend did not exit after SIGTERM; sending SIGKILL", flush=True)

    signal_process_group(process, signal.SIGKILL)
    process.wait()
    print(f"Backend exited with code {process.returncode}", flush=True)


class RuntimeState:
    def __init__(self, process: subprocess.Popen[bytes], stop_timeout: float) -> None:
        self.process = process
        self.stop_timeout = stop_timeout
        self.httpd: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._stopping = False
        self._stop_thread: threading.Thread | None = None

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    def request_stop(self, reason: str) -> threading.Thread | None:
        with self._lock:
            if self._stopping:
                return self._stop_thread
            self._stopping = True

        thread = threading.Thread(
            target=self._stop_backend_and_listener,
            args=(reason,),
            daemon=True,
        )
        with self._lock:
            self._stop_thread = thread
        thread.start()
        return thread

    def _stop_backend_and_listener(self, reason: str) -> None:
        stop_backend(self.process, self.stop_timeout, reason)
        if self.httpd is not None:
            self.httpd.shutdown()

    def shutdown_listener(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()

    def wait_for_stop(self) -> None:
        with self._lock:
            thread = self._stop_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()


def make_ack_handler(state: RuntimeState) -> type[BaseHTTPRequestHandler]:
    class AckHandler(BaseHTTPRequestHandler):
        server_version = "VLLMAckServer/1.0"

        def do_GET(self) -> None:
            self._handle_request()

        def do_POST(self) -> None:
            self._handle_request()

        def log_message(self, fmt: str, *args: object) -> None:
            print(
                f"{self.address_string()} - {fmt % args}",
                file=sys.stderr,
                flush=True,
            )

        def _handle_request(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(content_length) if content_length > 0 else b""

            if is_ack_request(self.path, body):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ACK received\n")
                state.request_stop("ACK received on listener")
                return

            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Expected ACK\n")

    return AckHandler


def format_command(command: list[str]) -> str:
    secret_flags = {"--api-key", "--hf-token"}
    redacted: list[str] = []
    redact_next = False

    for value in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue

        if value in secret_flags:
            redacted.append(value)
            redact_next = True
            continue

        if any(value.startswith(f"{flag}=") for flag in secret_flags):
            flag, _secret = value.split("=", 1)
            redacted.append(f"{flag}=<redacted>")
            continue

        redacted.append(value)

    return " ".join(redacted)


def start_backend(script: Path, backend_args: list[str]) -> subprocess.Popen[bytes]:
    command = ["bash", str(script), *backend_args]
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SCRIPT_DIR}{os.pathsep}{python_path}" if python_path else str(SCRIPT_DIR)
    )
    print(f"Starting backend: {format_command(command)}", flush=True)
    return subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        env=env,
        start_new_session=True,
    )


def monitor_backend(state: RuntimeState) -> None:
    return_code = state.process.wait()
    if state.stopping:
        return

    print(
        f"Backend exited with code {return_code}; stopping ACK listener",
        file=sys.stderr,
        flush=True,
    )
    state.shutdown_listener()


def main(argv: list[str]) -> int:
    args, backend_args = parse_args(argv)

    env_values = parse_dotenv(ENV_PATH)
    cpu_machine_url = env_values.get("CPU_MACHINE_URL")
    if not cpu_machine_url:
        raise RuntimeError(f"CPU_MACHINE_URL is not set in {ENV_PATH}")

    vllm_port = resolve_vllm_port(args.vllm_port, env_values)
    backend_args = make_backend_args(backend_args, args.model, vllm_port)

    backend_script = resolve_backend_script(args.backend_script)
    process = start_backend(backend_script, backend_args)
    state = RuntimeState(process, args.stop_timeout)

    try:
        def handle_signal(signum: int, _frame: object) -> None:
            state.request_stop(f"Received signal {signum}")

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        monitor_thread = threading.Thread(
            target=monitor_backend,
            args=(state,),
            daemon=True,
        )
        monitor_thread.start()

        wait_for_vllm_health(process, vllm_port)

        handler = make_ack_handler(state)
        httpd = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
        state.httpd = httpd

        if not send_ack(cpu_machine_url, args.ack_timeout, args.ack_retries):
            print(
                "Failed to send ACK to CPU machine; continuing to listen",
                file=sys.stderr,
            )

        print(
            f"Listening for ACK on {args.listen_host}:{args.listen_port}",
            flush=True,
        )

        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
            if process.poll() is None:
                state.request_stop("Server exiting")
                state.wait_for_stop()
    except Exception:
        if process.poll() is None:
            stop_backend(process, args.stop_timeout, "Server failed")
        raise

    return process.returncode if process.returncode is not None else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"server.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
