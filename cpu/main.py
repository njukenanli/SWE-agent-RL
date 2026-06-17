#!/usr/bin/env python3
from __future__ import annotations

import queue
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
ACK_PAYLOAD = "ACK"
DEFAULT_HOST = "0.0.0.0"
PLACEHOLDER_SLEEP_SECONDS = 60


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


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"CPU_MACHINE_PORT must be an integer, got: {value!r}") from exc

    if not 1 <= port <= 65535:
        raise ValueError(f"CPU_MACHINE_PORT must be in [1, 65535], got: {port}")

    return port


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


def send_ack(gpu_machine_url: str, timeout: float = 5.0) -> None:
    request = Request(
        gpu_machine_url,
        data=ACK_PAYLOAD.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )

    with urlopen(request, timeout=timeout) as response:
        response.read(256)
        print(f"Sent ACK to GPU machine, status={response.status}", flush=True)


def sleep_placeholder(stop_event: threading.Event) -> bool:
    return not stop_event.wait(PLACEHOLDER_SLEEP_SECONDS)


def drain_ack_queue(ack_queue: queue.Queue[None]) -> None:
    while True:
        try:
            ack_queue.get_nowait()
        except queue.Empty:
            return


def make_ack_handler(ack_queue: queue.Queue[None]) -> type[BaseHTTPRequestHandler]:
    class AckHandler(BaseHTTPRequestHandler):
        server_version = "CPUAckServer/1.0"

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
                ack_queue.put(None)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ACK received\n")
                return

            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Expected ACK\n")

    return AckHandler


def main() -> int:
    env_values = parse_dotenv(ENV_PATH)
    gpu_machine_url = env_values.get("GPU_MACHINE_URL")
    cpu_machine_port = env_values.get("CPU_MACHINE_PORT")

    if not gpu_machine_url:
        raise RuntimeError(f"GPU_MACHINE_URL is not set in {ENV_PATH}")
    if not cpu_machine_port:
        raise RuntimeError(f"CPU_MACHINE_PORT is not set in {ENV_PATH}")

    port = parse_port(cpu_machine_port)
    ack_queue: queue.Queue[None] = queue.Queue()
    httpd = ThreadingHTTPServer((DEFAULT_HOST, port), make_ack_handler(ack_queue))
    stop_event = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stop_event.set()
        httpd.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Listening for ACK on {DEFAULT_HOST}:{port}", flush=True)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        while not stop_event.is_set():
            drain_ack_queue(ack_queue)
            print("Waiting for ACK from GPU machine", flush=True)
            while not stop_event.is_set():
                try:
                    ack_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    continue
            if stop_event.is_set():
                break

            print(
                f"ACK received; sleeping {PLACEHOLDER_SLEEP_SECONDS} seconds",
                flush=True,
            )
            if not sleep_placeholder(stop_event):
                break

            try:
                send_ack(gpu_machine_url)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"Failed to send ACK to GPU machine: {exc}", file=sys.stderr)
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"main.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
