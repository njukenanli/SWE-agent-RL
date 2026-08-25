#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
import hashlib
import hmac
import json
import math
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
DEFAULT_LISTEN_PORT = 8004
DEFAULT_TRAJECTORY_UPLOAD_HOST = "127.0.0.1"
DAPO_JSON_PATH = SCRIPT_DIR.parent / "verl" / "dapo.json"
UPLOAD_CHUNK_SIZE = 1024 * 1024
ACK_PAYLOAD = "ACK"
TOKEN_DETAILS_MIDDLEWARE = "server.TokenDetailsDefaultsMiddleware"
TOKEN_DETAILS_ENDPOINTS = {
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/chat/completions/batch",
}
NEUTRAL_SAMPLING_PARAMS = {
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}


class TokenDetailsDefaultsMiddleware:
    """Inject token metadata and allow only temperature-based sampling."""

    def __init__(self, app: object) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Awaitable[dict[str, object]]],
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
                await self.app(
                    scope,
                    _single_message_receiver(message, receive),
                    send,
                )
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
                    {"type": "http.request", "body": original_body},
                    receive,
                ),
                send,
            )
            return

        if not isinstance(payload, dict):
            await self.app(
                scope,
                _single_message_receiver(
                    {"type": "http.request", "body": original_body},
                    receive,
                ),
                send,
            )
            return

        # Preserve request temperature: vLLM gives it precedence over server
        # generation defaults. Force every other sampling filter and penalty
        # to its neutral value so temperature is the only active control.
        payload.update(NEUTRAL_SAMPLING_PARAMS)
        # This service's consumers require aligned token IDs and probabilities,
        # so do not allow request defaults to disable the response metadata.
        payload["return_token_ids"] = True
        if path == "/v1/completions":
            payload["logprobs"] = 1
        else:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

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
                {"type": "http.request", "body": modified_body},
                receive,
            ),
            send,
        )


def _single_message_receiver(
    message: dict[str, object],
    receive_next: Callable[[], Awaitable[dict[str, object]]],
) -> Callable[[], Awaitable[dict[str, object]]]:
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return message
        return await receive_next()

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
        "--epoch",
        type=int,
        required=True,
        help="Zero-based rollout epoch expected in the uploaded DAPO data.",
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
        "--trajectory-upload-host",
        default=DEFAULT_TRAJECTORY_UPLOAD_HOST,
        help=(
            "Host for the trajectory upload listener. Default: "
            f"{DEFAULT_TRAJECTORY_UPLOAD_HOST} (SSH tunnel only)."
        ),
    )
    parser.add_argument(
        "--trajectory-upload-port",
        type=int,
        default=None,
        help="Upload listener port. If omitted, TRAJECTORY_UPLOAD_PORT is required in ../.env.",
    )
    parser.add_argument(
        "--dapo-json",
        type=Path,
        default=DAPO_JSON_PATH,
        help=f"DAPO JSON destination. Default: {DAPO_JSON_PATH}",
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
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ack-retry-interval",
        type=float,
        default=60.0,
        help="Seconds between CPU readiness ACK attempts. Default: 60.",
    )
    parser.add_argument(
        "--stop-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait after SIGINT before escalating to SIGTERM. Default: 30.0",
    )

    args, backend_args = parser.parse_known_args(argv)
    if args.epoch < 0:
        parser.error("--epoch must be non-negative")
    if args.ack_retry_interval < 0:
        parser.error("--ack-retry-interval must be non-negative")
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


def resolve_trajectory_upload_port(explicit_port: int | None, env_values: dict[str, str]) -> int:
    if explicit_port is not None:
        return validate_port(explicit_port, "--trajectory-upload-port")

    env_port = env_values.get("TRAJECTORY_UPLOAD_PORT")
    if not env_port:
        raise RuntimeError(f"TRAJECTORY_UPLOAD_PORT is not set in {ENV_PATH}")

    try:
        return validate_port(int(env_port), "TRAJECTORY_UPLOAD_PORT")
    except ValueError as exc:
        raise ValueError(f"invalid TRAJECTORY_UPLOAD_PORT in {ENV_PATH}: {env_port!r}") from exc


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


def send_ack_until_received(
    cpu_machine_url: str,
    timeout: float,
    retry_interval: float,
    *,
    epoch: int,
    shutdown_event: threading.Event,
    process: subprocess.Popen[bytes],
) -> bool:
    if retry_interval < 0:
        raise ValueError("ACK retry interval must be non-negative")

    payload = ACK_PAYLOAD.encode("utf-8")
    attempt = 0

    while not shutdown_event.is_set() and process.poll() is None:
        attempt += 1
        request = Request(
            cpu_machine_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "X-DAPO-Epoch": str(epoch),
            },
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                response.read(256)
                if response.status == 200:
                    print(
                        f"CPU machine received readiness ACK, status=200, attempt={attempt}",
                        flush=True,
                    )
                    return True
                failure = f"unexpected HTTP status {response.status}"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            failure = str(exc)

        if shutdown_event.is_set() or process.poll() is not None:
            break
        print(
            f"CPU readiness ACK attempt {attempt} failed: {failure}; "
            f"retrying in {retry_interval:g} seconds",
            file=sys.stderr,
            flush=True,
        )
        if shutdown_event.wait(retry_interval):
            break

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


def validate_dapo_data(data: object) -> None:
    if not isinstance(data, list) or not data:
        raise ValueError("DAPO data must be a non-empty list of task groups")

    for group_index, group in enumerate(data):
        if not isinstance(group, list) or not group:
            raise ValueError(f"group {group_index} must be a non-empty list of trajectories")
        for trajectory_index, trajectory in enumerate(group):
            if not isinstance(trajectory, list):
                raise ValueError(f"group {group_index}, trajectory {trajectory_index} must be a list")

            reward: float | None = None
            for step_index, step in enumerate(trajectory):
                label = f"group {group_index}, trajectory {trajectory_index}, step {step_index}"
                if not isinstance(step, list) or len(step) != 3:
                    raise ValueError(f"{label} must be [input_token_ids, output_pairs, reward]")
                input_token_ids, output_pairs, step_reward = step

                if not isinstance(input_token_ids, list) or any(
                    isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in input_token_ids
                ):
                    raise ValueError(f"{label} input_token_ids must be a list of integers")
                if not isinstance(output_pairs, list):
                    raise ValueError(f"{label} output_pairs must be a list")
                for pair_index, pair in enumerate(output_pairs):
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError(f"{label}, output {pair_index} must be [token_id, probability]")
                    token_id, probability = pair
                    if isinstance(token_id, bool) or not isinstance(token_id, int):
                        raise ValueError(f"{label}, output {pair_index} token ID must be an integer")
                    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                        raise ValueError(f"{label}, output {pair_index} probability must be numeric")
                    try:
                        probability = float(probability)
                    except OverflowError as exc:
                        raise ValueError(
                            f"{label}, output {pair_index} probability must be finite"
                        ) from exc
                    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                        raise ValueError(f"{label}, output {pair_index} probability must be in [0, 1]")

                if isinstance(step_reward, bool) or not isinstance(step_reward, (int, float)):
                    raise ValueError(f"{label} reward must be 0 or 1")
                try:
                    step_reward = float(step_reward)
                except OverflowError as exc:
                    raise ValueError(f"{label} reward must be 0 or 1") from exc
                if not math.isfinite(step_reward) or step_reward not in {0.0, 1.0}:
                    raise ValueError(f"{label} reward must be 0 or 1")
                if reward is None:
                    reward = step_reward
                elif step_reward != reward:
                    raise ValueError(
                        f"group {group_index}, trajectory {trajectory_index} has inconsistent rewards"
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
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        stop_timeout: float,
        *,
        epoch: int,
        dapo_path: Path,
    ) -> None:
        self.process = process
        self.stop_timeout = stop_timeout
        self.epoch = epoch
        self.dapo_path = dapo_path
        self.httpd: ThreadingHTTPServer | None = None
        self.upload_httpd: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self._stopping = False
        self._normal_completion = False
        self._stop_thread: threading.Thread | None = None
        self._ready_epoch: int | None = None
        self._ready_checksum: str | None = None

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    @property
    def normal_completion(self) -> bool:
        with self._lock:
            return self._normal_completion

    @property
    def upload_lock(self) -> threading.Lock:
        return self._upload_lock

    def mark_trajectory_ready(self, epoch: int, checksum: str) -> None:
        with self._lock:
            self._ready_epoch = epoch
            self._ready_checksum = checksum

    def trajectory_is_ready(self) -> bool:
        with self._lock:
            return self._ready_epoch == self.epoch and self._ready_checksum is not None

    def request_stop(
        self, reason: str, *, normal_completion: bool = False
    ) -> threading.Thread | None:
        with self._lock:
            if normal_completion:
                if self._ready_epoch != self.epoch or self._ready_checksum is None:
                    raise RuntimeError(
                        f"Cannot complete epoch {self.epoch} before its DAPO data is received"
                    )
                self._normal_completion = True
            if self._stopping:
                return self._stop_thread
            self._stopping = True
            self.shutdown_event.set()

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
        if self.upload_httpd is not None:
            self.upload_httpd.shutdown()

    def shutdown_listener(self) -> None:
        self.shutdown_event.set()
        if self.httpd is not None:
            self.httpd.shutdown()
        if self.upload_httpd is not None:
            self.upload_httpd.shutdown()

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
                epoch_header = self.headers.get("X-DAPO-Epoch")
                try:
                    ack_epoch = state.epoch if epoch_header is None else int(epoch_header)
                except ValueError:
                    response = b"X-DAPO-Epoch must be an integer\n"
                    self.send_response(400)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if ack_epoch < 0 or ack_epoch > state.epoch:
                    response = f"Expected ACK epoch at most {state.epoch}, got {ack_epoch}\n".encode()
                    self.send_response(409)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if ack_epoch < state.epoch:
                    response = f"Epoch {ack_epoch} was already acknowledged\n".encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if not state.trajectory_is_ready():
                    response = f"DAPO data for epoch {state.epoch} has not been received\n".encode()
                    self.send_response(409)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b"ACK received\n")))
                self.end_headers()
                self.wfile.write(b"ACK received\n")
                state.request_stop("ACK received on listener", normal_completion=True)
                return

            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Expected ACK\n")

    return AckHandler


def make_trajectory_upload_handler(state: RuntimeState) -> type[BaseHTTPRequestHandler]:
    class TrajectoryUploadHandler(BaseHTTPRequestHandler):
        server_version = "DAPOUploadServer/1.0"

        def do_POST(self) -> None:
            self._handle_upload()

        def do_PUT(self) -> None:
            self._handle_upload()

        def log_message(self, fmt: str, *args: object) -> None:
            print(
                f"{self.address_string()} - {fmt % args}",
                file=sys.stderr,
                flush=True,
            )

        def _send_text(self, status: int, message: str) -> None:
            body = message.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_upload(self) -> None:
            if urlparse(self.path).path.rstrip("/") != "/dapo":
                self._send_text(404, "Expected /dapo\n")
                return
            if "application/json" not in self.headers.get("Content-Type", "").lower():
                self._send_text(415, "Expected Content-Type: application/json\n")
                return

            try:
                upload_epoch = int(self.headers.get("X-DAPO-Epoch", ""))
            except ValueError:
                self._send_text(400, "X-DAPO-Epoch must be an integer\n")
                return
            if upload_epoch != state.epoch:
                self._send_text(409, f"Expected epoch {state.epoch}, got {upload_epoch}\n")
                return

            expected_checksum = self.headers.get("X-DAPO-SHA256", "").lower()
            if len(expected_checksum) != 64 or any(
                character not in "0123456789abcdef" for character in expected_checksum
            ):
                self._send_text(400, "X-DAPO-SHA256 must be a hexadecimal SHA-256 digest\n")
                return

            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_text(411, "A valid Content-Length is required\n")
                return
            if content_length <= 0:
                self._send_text(411, "Content-Length must be positive\n")
                return

            if not state.upload_lock.acquire(blocking=False):
                self._send_text(409, "Another DAPO upload is in progress\n")
                return

            incoming_dir = state.dapo_path.parent / ".incoming"
            temporary_path = incoming_dir / f"dapo.epoch_{upload_epoch}.{threading.get_ident()}.tmp"
            try:
                incoming_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                remaining = content_length
                with temporary_path.open("wb") as output_file:
                    while remaining:
                        chunk = self.rfile.read(min(UPLOAD_CHUNK_SIZE, remaining))
                        if not chunk:
                            raise ValueError(
                                f"Upload ended early with {remaining} of {content_length} bytes missing"
                            )
                        output_file.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    output_file.flush()
                    os.fsync(output_file.fileno())

                actual_checksum = digest.hexdigest()
                if not hmac.compare_digest(actual_checksum, expected_checksum):
                    raise ValueError(
                        f"SHA-256 mismatch: expected {expected_checksum}, got {actual_checksum}"
                    )

                try:
                    with temporary_path.open("r", encoding="utf-8") as input_file:
                        dapo_data = json.load(input_file)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Uploaded file is not valid UTF-8 DAPO JSON") from exc
                validate_dapo_data(dapo_data)

                state.dapo_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_path, state.dapo_path)
                directory_fd = os.open(state.dapo_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                state.mark_trajectory_ready(upload_epoch, actual_checksum)
                print(
                    f"Received epoch {upload_epoch} DAPO data at {state.dapo_path}, "
                    f"bytes={content_length}, sha256={actual_checksum}",
                    flush=True,
                )
                self._send_text(201, "DAPO data received\n")
            except ValueError as exc:
                self._send_text(400, f"Invalid DAPO upload: {exc}\n")
            except OSError as exc:
                self._send_text(500, f"Failed to store DAPO upload: {exc}\n")
            finally:
                temporary_path.unlink(missing_ok=True)
                state.upload_lock.release()

    return TrajectoryUploadHandler


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


def server_exit_code(state: RuntimeState) -> int:
    """Return success only after the current epoch's upload and ACK handshake."""
    if state.normal_completion:
        return 0

    # vLLM may return zero after a graceful but unexpected shutdown. That does
    # not complete this workflow: without the current DAPO upload and CPU ACK,
    # main.sh must stop instead of training on a stale dapo.json.
    return_code = state.process.returncode
    return return_code if return_code not in {None, 0} else 1


def main(argv: list[str]) -> int:
    args, backend_args = parse_args(argv)

    env_values = parse_dotenv(ENV_PATH)
    cpu_machine_url = env_values.get("CPU_MACHINE_URL")
    if not cpu_machine_url:
        raise RuntimeError(f"CPU_MACHINE_URL is not set in {ENV_PATH}")

    vllm_port = resolve_vllm_port(args.vllm_port, env_values)
    trajectory_upload_port = resolve_trajectory_upload_port(args.trajectory_upload_port, env_values)
    dapo_path = args.dapo_json.expanduser().resolve()
    backend_args = make_backend_args(backend_args, args.model, vllm_port)

    backend_script = resolve_backend_script(args.backend_script)
    process = start_backend(backend_script, backend_args)
    state = RuntimeState(process, args.stop_timeout, epoch=args.epoch, dapo_path=dapo_path)

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

        if not send_ack_until_received(
            cpu_machine_url,
            args.ack_timeout,
            args.ack_retry_interval,
            epoch=args.epoch,
            shutdown_event=state.shutdown_event,
            process=process,
        ):
            state.wait_for_stop()
            if process.poll() is None:
                stop_backend(process, args.stop_timeout, "ACK retry loop stopped")
            return server_exit_code(state)

        handler = make_ack_handler(state)
        httpd = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
        state.httpd = httpd
        upload_handler = make_trajectory_upload_handler(state)
        upload_httpd = ThreadingHTTPServer(
            (args.trajectory_upload_host, trajectory_upload_port),
            upload_handler,
        )
        state.upload_httpd = upload_httpd

        print(
            f"Listening for ACK on {args.listen_host}:{args.listen_port}",
            flush=True,
        )
        print(
            f"Listening for epoch {args.epoch} DAPO upload on "
            f"{args.trajectory_upload_host}:{trajectory_upload_port}",
            flush=True,
        )
        upload_thread = threading.Thread(target=upload_httpd.serve_forever, daemon=True)
        upload_thread.start()

        try:
            httpd.serve_forever()
        finally:
            if process.poll() is None:
                state.request_stop("Server exiting")
            state.wait_for_stop()
            httpd.server_close()
            upload_httpd.server_close()
    except Exception:
        if process.poll() is None:
            stop_backend(process, args.stop_timeout, "Server failed")
        raise

    return server_exit_code(state)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"server.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
