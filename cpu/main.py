#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
SWEAGENT_DIR = SCRIPT_DIR / "sweagent"
LOG_PATH = SCRIPT_DIR / "log.out"
ACK_PAYLOAD = "ACK"
DEFAULT_HOST = "0.0.0.0"


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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {parsed}")
    return parsed


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"dataset file does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel_instances",
        type=positive_int,
        required=True,
        help="Number of SWE-agent task instances to run in parallel.",
    )
    parser.add_argument(
        "--dataset",
        type=existing_file,
        required=True,
        help="Path to the SWE-bench JSONL dataset.",
    )
    return parser.parse_args(argv)


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


def run_sweagent(
    *,
    parallel_instances: int,
    dataset: Path,
    iteration_num: int,
    llm_api_key: str,
    llm_api_base: str,
) -> None:
    epoch = iteration_num - 1
    subprocess_env = os.environ.copy()
    subprocess_env["LLM_API_KEY"] = llm_api_key
    subprocess_env["LLM_API_BASE"] = llm_api_base
    common_args = [
        "--parallel_instances",
        str(parallel_instances),
        "--instances.type",
        "swe_bench",
        "--instances.subset",
        str(dataset),
        "--epoch",
        str(epoch),
    ]
    commands = [
        ["sweagent", "run-batch", "--config", "config/test.yaml", *common_args],
        ["sweagent", "run-batch", "--config", "config/train.yaml", *common_args],
    ]

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        for command in commands:
            print(
                f"Running {' '.join(command)}; output redirected to {LOG_PATH}",
                flush=True,
            )
            subprocess.run(
                command,
                cwd=SWEAGENT_DIR,
                stdout=log_file,
                stderr=log_file,
                env=subprocess_env,
                check=True,
            )
            log_file.flush()


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_values = parse_dotenv(ENV_PATH)
    gpu_machine_url = env_values.get("GPU_MACHINE_URL")
    cpu_machine_port = env_values.get("CPU_MACHINE_PORT")
    llm_api_key = env_values.get("LLM_API_KEY")
    llm_api_base = env_values.get("LLM_API_BASE")

    if not gpu_machine_url:
        raise RuntimeError(f"GPU_MACHINE_URL is not set in {ENV_PATH}")
    if not cpu_machine_port:
        raise RuntimeError(f"CPU_MACHINE_PORT is not set in {ENV_PATH}")
    if not llm_api_key:
        raise RuntimeError(f"LLM_API_KEY is not set in {ENV_PATH}")
    if not llm_api_base:
        raise RuntimeError(f"LLM_API_BASE is not set in {ENV_PATH}")

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
    iteration_num = 1

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

            print(f"ACK received; starting iteration {iteration_num}", flush=True)
            run_sweagent(
                parallel_instances=args.parallel_instances,
                dataset=args.dataset,
                iteration_num=iteration_num,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
            )

            try:
                send_ack(gpu_machine_url)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"Failed to send ACK to GPU machine: {exc}", file=sys.stderr)
            else:
                iteration_num += 1
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
