#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from ruamel.yaml import YAML


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
SWEAGENT_DIR = SCRIPT_DIR / "sweagent"
LOG_PATH = SCRIPT_DIR / "log.out"
TEMP_DATASET_PATH = SWEAGENT_DIR / "temp.jsonl"
ACK_PAYLOAD = "ACK"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_ACK_RETRY_INTERVAL = 60.0
DEFAULT_TRAJECTORY_UPLOAD_TIMEOUT = 60 * 60
DEFAULT_TRAJECTORY_UPLOAD_RETRIES = 5
DEFAULT_TRAJECTORY_UPLOAD_RETRY_INTERVAL = 15.0
UPLOAD_CHUNK_SIZE = 1024 * 1024


class _RetryableUploadError(RuntimeError):
    """A temporary HTTP response that should retry the DAPO upload."""


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


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got: {parsed}")
    return parsed


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"dataset file does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers",
        type=positive_int,
        required=True,
        help="Maximum number of SWE-agent rollouts allowed to run concurrently.",
    )
    parser.add_argument(
        "--train_set",
        type=existing_file,
        required=True,
        help="Path to the training JSONL dataset from which each epoch batch is selected.",
    )
    parser.add_argument(
        "--test_set",
        type=existing_file,
        required=True,
        help="Path to the complete test JSONL dataset used in every epoch.",
    )
    parser.add_argument(
        "--batch_size",
        type=positive_int,
        required=True,
        help="Number of dataset instances to use for each epoch.",
    )
    parser.add_argument(
        "--start_epoch",
        type=non_negative_int,
        default=0,
        help="First zero-based epoch to run. Default: 0.",
    )
    parser.add_argument(
        "--ack-retry-interval",
        type=float,
        default=DEFAULT_ACK_RETRY_INTERVAL,
        help="Seconds between GPU completion ACK attempts. Default: 60.",
    )
    args = parser.parse_args(argv)
    if args.ack_retry_interval < 0:
        parser.error("--ack-retry-interval must be non-negative")
    return args


def write_epoch_dataset_batch(
    dataset: Path,
    *,
    current_epoch: int,
    batch_size: int,
    output_path: Path = TEMP_DATASET_PATH,
) -> Path:
    if current_epoch < 0:
        raise ValueError(f"current_epoch must be non-negative, got: {current_epoch}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")

    records = [line for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset_size = len(records)
    if dataset_size == 0:
        raise ValueError(f"dataset contains no instances: {dataset}")
    if batch_size > dataset_size:
        raise ValueError(
            f"batch_size ({batch_size}) cannot exceed dataset size ({dataset_size})"
        )

    start = (current_epoch * batch_size) % dataset_size
    end = ((current_epoch + 1) * batch_size) % dataset_size
    if start < end:
        batch = records[start:end]
    elif start > end:
        batch = records[start:] + records[:end]
    else:
        # With batch_size <= dataset_size, equal modular boundaries mean one
        # complete pass through the dataset.
        batch = records[:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_path = output_path.with_name(f".{output_path.name}.tmp")
    incoming_path.write_text("".join(f"{record}\n" for record in batch), encoding="utf-8")
    os.replace(incoming_path, output_path)
    print(
        f"Prepared epoch {current_epoch} dataset batch with {len(batch)} instance(s) "
        f"from [{start}, {end}) at {output_path}",
        flush=True,
    )
    return output_path


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


def send_ack_until_received(
    gpu_machine_url: str,
    *,
    epoch: int,
    shutdown_event: threading.Event,
    timeout: float = 5.0,
    retry_interval: float = DEFAULT_ACK_RETRY_INTERVAL,
) -> bool:
    if retry_interval < 0:
        raise ValueError("ACK retry interval must be non-negative")

    payload = ACK_PAYLOAD.encode("utf-8")
    attempt = 0
    while not shutdown_event.is_set():
        attempt += 1
        request = Request(
            gpu_machine_url,
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
                        f"GPU machine received epoch {epoch} completion ACK, "
                        f"status=200, attempt={attempt}",
                        flush=True,
                    )
                    return True
                failure = f"unexpected HTTP status {response.status}"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            failure = str(exc)

        if shutdown_event.is_set():
            break
        print(
            f"GPU completion ACK attempt {attempt} for epoch {epoch} failed: {failure}; "
            f"retrying in {retry_interval:g} seconds",
            file=sys.stderr,
            flush=True,
        )
        if shutdown_event.wait(retry_interval):
            break

    return False


def get_success(model: str, epoch: int, mode: Literal["train", "test"]) -> tuple[int, int, float]:
    trajectory_dir = SWEAGENT_DIR / "logs" / model / str(epoch) / mode
    trajectory_paths: list[Path] = []
    if trajectory_dir.is_dir():
        for instance_dir in trajectory_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            for sample_dir in instance_dir.iterdir():
                if sample_dir.is_dir() and sample_dir.name.isdecimal():
                    trajectory_paths.append(sample_dir / f"{instance_dir.name}.traj")

    total = len(trajectory_paths)
    success = 0

    for trajectory_path in trajectory_paths:
        try:
            trajectory_data = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Empty and malformed trajectories still count as executed samples,
            # but cannot count as successful ones.
            continue

        if (
            isinstance(trajectory_data, dict)
            and isinstance(trajectory_data.get("info"), dict)
            and trajectory_data["info"].get("success") is True
        ):
            success += 1

    success_rate = round(success / total * 100, 2) if total else 0.0
    print(
        f"Epoch: {epoch}, Num of successful samples: {success}, "
        f"Num of all samples executed: {total}, Success rate: {success_rate}",
        flush=True,
    )
    return success, total, success_rate


def _get_model_name(config_path: Path) -> str:
    config = YAML(typ="safe").load(config_path.read_text(encoding="utf-8"))
    try:
        model = config["agent"]["model"]["name"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Could not find agent.model.name in {config_path}") from exc
    if not isinstance(model, str) or not model:
        raise ValueError(f"agent.model.name in {config_path} must be a non-empty string")
    return model


def _get_sample_count(config_path: Path) -> int:
    config = YAML(typ="safe").load(config_path.read_text(encoding="utf-8"))
    try:
        samples = config["agent"]["model"]["samples"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Could not find agent.model.samples in {config_path}") from exc
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"agent.model.samples in {config_path} must be a positive integer")
    return samples


def _read_token_ids(value: object, field: str, trajectory_path: Path, step_index: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{trajectory_path}: trajectory step {step_index} {field} must be a list")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in value):
        raise ValueError(f"{trajectory_path}: trajectory step {step_index} {field} must contain integers")
    return value


def _read_probabilities(value: object, trajectory_path: Path, step_index: int) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(
            f"{trajectory_path}: trajectory step {step_index} output_token_probabilities must be a list"
        )

    probabilities: list[float] = []
    for probability in value:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(
                f"{trajectory_path}: trajectory step {step_index} "
                "output_token_probabilities must contain numbers"
            )
        try:
            probability = float(probability)
        except OverflowError as exc:
            raise ValueError(
                f"{trajectory_path}: trajectory step {step_index} has an invalid probability"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{trajectory_path}: trajectory step {step_index} has probability outside [0, 1]: "
                f"{probability}"
            )
        probabilities.append(probability)
    return probabilities


def _convert_trajectory(trajectory_path: Path) -> list[list[object]]:
    try:
        trajectory_data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read trajectory JSON: {trajectory_path}") from exc
    if not isinstance(trajectory_data, dict):
        raise ValueError(f"{trajectory_path}: trajectory file must contain a JSON object")

    info = trajectory_data.get("info")
    if not isinstance(info, dict) or not isinstance(info.get("success"), bool):
        raise ValueError(f"{trajectory_path}: info.success must be a boolean")
    reward = int(info["success"])

    trajectory = trajectory_data.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise ValueError(f"{trajectory_path}: trajectory must be a list")

    converted: list[list[object]] = []
    for step_index, step in enumerate(trajectory, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"{trajectory_path}: trajectory step {step_index} must be an object")

        input_token_ids = _read_token_ids(
            step.get("input_token_ids"), "input_token_ids", trajectory_path, step_index
        )
        output_token_ids = _read_token_ids(
            step.get("output_token_ids"), "output_token_ids", trajectory_path, step_index
        )
        probabilities = _read_probabilities(
            step.get("output_token_probabilities"), trajectory_path, step_index
        )
        if len(output_token_ids) != len(probabilities):
            raise ValueError(
                f"{trajectory_path}: trajectory step {step_index} has "
                f"{len(output_token_ids)} output token IDs but {len(probabilities)} probabilities"
            )

        output_pairs = [list(pair) for pair in zip(output_token_ids, probabilities, strict=True)]
        converted.append([input_token_ids, output_pairs, reward])
    return converted


def collect_dapo_trajectories(*, model: str, epoch: int, samples: int) -> list[list[list[list[object]]]]:
    model_path = Path(model)
    if model_path.is_absolute() or ".." in model_path.parts:
        raise ValueError(f"Model name must be a safe relative path, got {model!r}")

    train_dir = SWEAGENT_DIR / "logs" / model_path / str(epoch) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Current epoch train trajectory directory does not exist: {train_dir}")

    instance_dirs = sorted(path for path in train_dir.iterdir() if path.is_dir())
    if not instance_dirs:
        raise ValueError(f"No training instances found in {train_dir}")

    expected_sample_ids = set(range(samples))
    groups: list[list[list[list[object]]]] = []
    for instance_dir in instance_dirs:
        sample_dirs = [path for path in instance_dir.iterdir() if path.is_dir() and path.name.isdecimal()]
        actual_sample_ids = {int(path.name) for path in sample_dirs}
        unexpected_sample_ids = actual_sample_ids - expected_sample_ids
        if unexpected_sample_ids:
            raise ValueError(
                f"{instance_dir}: found unexpected sample directories "
                f"{sorted(unexpected_sample_ids)}; expected IDs 0..{samples - 1}"
            )

        group: list[list[list[object]]] = []
        for sample_id in range(samples):
            trajectory_path = instance_dir / str(sample_id) / f"{instance_dir.name}.traj"
            if not trajectory_path.is_file():
                print(
                    f"Missing sample trajectory {trajectory_path}; inserting an empty reward-0 placeholder",
                    file=sys.stderr,
                    flush=True,
                )
                group.append([])
                continue
            group.append(_convert_trajectory(trajectory_path))
        groups.append(group)
    return groups


def write_dapo_json(*, model: str, epoch: int, samples: int) -> Path:
    train_dir = SWEAGENT_DIR / "logs" / model / str(epoch) / "train"
    dapo_path = train_dir / "dapo.json"
    temporary_path = train_dir / ".dapo.json.tmp"
    groups = collect_dapo_trajectories(model=model, epoch=epoch, samples=samples)

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(groups, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, dapo_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    trajectory_count = sum(len(group) for group in groups)
    step_count = sum(len(trajectory) for group in groups for trajectory in group)
    print(
        f"Wrote DAPO data for epoch {epoch} to {dapo_path}: "
        f"groups={len(groups)}, trajectories={trajectory_count}, steps={step_count}",
        flush=True,
    )
    return dapo_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(UPLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def upload_dapo_json(
    upload_url: str,
    dapo_path: Path,
    *,
    epoch: int,
    timeout: float = DEFAULT_TRAJECTORY_UPLOAD_TIMEOUT,
    retries: int = DEFAULT_TRAJECTORY_UPLOAD_RETRIES,
    retry_interval: float = DEFAULT_TRAJECTORY_UPLOAD_RETRY_INTERVAL,
) -> None:
    parsed = urlparse(upload_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"TRAJECTORY_UPLOAD_URL must be an HTTP(S) URL, got {upload_url!r}")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("TRAJECTORY_UPLOAD_URL must not contain credentials or a fragment")
    if timeout <= 0:
        raise ValueError("trajectory upload timeout must be positive")
    if retry_interval < 0:
        raise ValueError("trajectory upload retry interval must be non-negative")

    request_path = parsed.path if parsed.path not in {"", "/"} else "/dapo"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    file_size = dapo_path.stat().st_size
    checksum = _sha256(dapo_path)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection

    retries = max(retries, 1)
    for attempt in range(1, retries + 1):
        connection = connection_type(parsed.hostname, port, timeout=timeout)
        try:
            connection.putrequest("PUT", request_path)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(file_size))
            connection.putheader("X-DAPO-Epoch", str(epoch))
            connection.putheader("X-DAPO-SHA256", checksum)
            connection.endheaders()
            with dapo_path.open("rb") as input_file:
                while chunk := input_file.read(UPLOAD_CHUNK_SIZE):
                    connection.send(chunk)

            response = connection.getresponse()
            response_body = response.read(4096).decode("utf-8", errors="replace").strip()
            if 200 <= response.status < 300:
                print(
                    f"Uploaded epoch {epoch} DAPO data to {upload_url}, "
                    f"bytes={file_size}, status={response.status}",
                    flush=True,
                )
                return
            message = f"trajectory upload returned HTTP {response.status}: {response_body or response.reason}"
            if response.status in {408, 425, 429} or 500 <= response.status < 600:
                raise _RetryableUploadError(message)
            raise RuntimeError(message)
        except (http.client.HTTPException, OSError, _RetryableUploadError) as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"Failed to upload epoch {epoch} DAPO data after {retries} attempt(s): {exc}"
                ) from exc
            print(
                f"Trajectory upload attempt {attempt}/{retries} failed: {exc}; "
                f"retrying in {retry_interval:g} seconds",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_interval)
        finally:
            connection.close()


def run_sweagent(
    *,
    num_workers: int,
    test_dataset: Path,
    train_dataset: Path,
    iteration_num: int,
    llm_api_key: str,
    llm_api_base: str,
) -> None:
    epoch = iteration_num - 1
    subprocess_env = os.environ.copy()
    subprocess_env["LLM_API_KEY"] = llm_api_key
    subprocess_env["LLM_API_BASE"] = llm_api_base
    common_args = [
        "--num_workers",
        str(num_workers),
        "--instances.type",
        "swe_bench",
        "--epoch",
        str(epoch),
    ]
    jobs: list[tuple[Literal["train", "test"], Path, Path]] = [
        ("test", Path("config/test.yaml"), test_dataset),
        ("train", Path("config/train.yaml"), train_dataset),
    ]

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        for mode, config_path, mode_dataset in jobs:
            model = _get_model_name(SWEAGENT_DIR / config_path)
            command = [
                "sweagent",
                "run-batch",
                "--config",
                str(config_path),
                *common_args,
                "--instances.subset",
                str(mode_dataset),
            ]
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
            get_success(model, epoch, mode)


def make_ack_handler(ack_queue: queue.Queue[int]) -> type[BaseHTTPRequestHandler]:
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
                try:
                    epoch = int(self.headers.get("X-DAPO-Epoch", ""))
                except ValueError:
                    response = b"X-DAPO-Epoch must be an integer\n"
                    self.send_response(400)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if epoch < 0:
                    response = b"X-DAPO-Epoch must be non-negative\n"
                    self.send_response(400)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return

                ack_queue.put(epoch)
                response = f"ACK received for epoch {epoch}\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
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
    trajectory_upload_url = env_values.get("TRAJECTORY_UPLOAD_URL")

    if not gpu_machine_url:
        raise RuntimeError(f"GPU_MACHINE_URL is not set in {ENV_PATH}")
    if not cpu_machine_port:
        raise RuntimeError(f"CPU_MACHINE_PORT is not set in {ENV_PATH}")
    if not llm_api_key:
        raise RuntimeError(f"LLM_API_KEY is not set in {ENV_PATH}")
    if not llm_api_base:
        raise RuntimeError(f"LLM_API_BASE is not set in {ENV_PATH}")
    if not trajectory_upload_url:
        raise RuntimeError(f"TRAJECTORY_UPLOAD_URL is not set in {ENV_PATH}")

    port = parse_port(cpu_machine_port)
    train_config_path = SWEAGENT_DIR / "config/train.yaml"
    train_model = _get_model_name(train_config_path)
    train_samples = _get_sample_count(train_config_path)
    ack_queue: queue.Queue[int] = queue.Queue()
    pending_ack_epochs: set[int] = set()
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
    iteration_num = args.start_epoch + 1

    try:
        while not stop_event.is_set():
            expected_epoch = iteration_num - 1
            print(f"Waiting for GPU readiness ACK for epoch {expected_epoch}", flush=True)
            while expected_epoch not in pending_ack_epochs and not stop_event.is_set():
                try:
                    ack_epoch = ack_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if ack_epoch < expected_epoch:
                    print(
                        f"Ignoring stale GPU readiness ACK for epoch {ack_epoch}; "
                        f"waiting for epoch {expected_epoch}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    pending_ack_epochs.add(ack_epoch)
            if stop_event.is_set():
                break
            pending_ack_epochs.remove(expected_epoch)

            print(f"ACK received; starting iteration {iteration_num}", flush=True)
            epoch_dataset = write_epoch_dataset_batch(
                args.train_set,
                current_epoch=expected_epoch,
                batch_size=args.batch_size,
            )
            run_sweagent(
                num_workers=args.num_workers,
                test_dataset=args.test_set,
                train_dataset=epoch_dataset,
                iteration_num=iteration_num,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
            )
            epoch = iteration_num - 1
            dapo_path = write_dapo_json(model=train_model, epoch=epoch, samples=train_samples)
            upload_dapo_json(trajectory_upload_url, dapo_path, epoch=epoch)

            if not send_ack_until_received(
                gpu_machine_url,
                epoch=epoch,
                shutdown_event=stop_event,
                retry_interval=args.ack_retry_interval,
            ):
                break
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
