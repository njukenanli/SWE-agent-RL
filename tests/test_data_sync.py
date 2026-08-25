from __future__ import annotations

import asyncio
import importlib.util
import json
import queue
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_main = _load_module("rl_cpu_main", ROOT / "cpu/main.py")
gpu_server = _load_module("rl_gpu_server", ROOT / "gpu/vllm/server.py")


def _run_token_details_middleware(payload):
    captured_payloads = []

    async def downstream(_scope, receive, _send):
        message = await receive()
        captured_payloads.append(json.loads(message["body"]))

    middleware = gpu_server.TokenDetailsDefaultsMiddleware(downstream)
    request_messages = iter(
        [{"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}]
    )

    async def receive():
        return next(request_messages)

    async def send(_message):
        return None

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
    )
    return captured_payloads[0]


def test_default_epoch_dataset_batch_path_is_inside_sweagent():
    assert cpu_main.TEMP_DATASET_PATH == ROOT / "cpu/sweagent/temp.jsonl"


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpu_token_details_middleware_preserves_request_temperature(temperature):
    payload = _run_token_details_middleware(
        {
            "model": "my-qwen-model",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": temperature,
            "top_p": 0.25,
            "top_k": 10,
            "min_p": 0.1,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.2,
        }
    )

    assert payload["temperature"] == temperature
    assert {
        key: payload[key] for key in gpu_server.NEUTRAL_SAMPLING_PARAMS
    } == gpu_server.NEUTRAL_SAMPLING_PARAMS
    assert payload["return_token_ids"] is True
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 1


def test_gpu_token_details_middleware_waits_for_real_disconnect():
    request_messages = asyncio.Queue()
    request_messages.put_nowait(
        {
            "type": "http.request",
            "body": json.dumps(
                {
                    "model": "my-qwen-model",
                    "messages": [{"role": "user", "content": "test"}],
                }
            ).encode(),
            "more_body": False,
        }
    )
    received_messages = []

    async def receive():
        return await request_messages.get()

    async def downstream(_scope, replay_receive, _send):
        received_messages.append(await replay_receive())
        disconnect_task = asyncio.create_task(replay_receive())
        await asyncio.sleep(0)
        assert not disconnect_task.done()

        request_messages.put_nowait({"type": "http.disconnect"})
        received_messages.append(await asyncio.wait_for(disconnect_task, timeout=1))

    async def send(_message):
        return None

    async def run_middleware():
        middleware = gpu_server.TokenDetailsDefaultsMiddleware(downstream)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )

    asyncio.run(run_middleware())

    assert received_messages[0]["type"] == "http.request"
    assert received_messages[1] == {"type": "http.disconnect"}


def test_cpu_cli_and_sweagent_command_use_num_workers(tmp_path, monkeypatch):
    train_set = tmp_path / "train.jsonl"
    test_set = tmp_path / "test.jsonl"
    train_set.write_text('{"instance_id":"train-task-0"}\n', encoding="utf-8")
    test_set.write_text('{"instance_id":"test-task-0"}\n', encoding="utf-8")
    args = cpu_main.parse_args(
        [
            "--num_workers",
            "7",
            "--train_set",
            str(train_set),
            "--test_set",
            str(test_set),
            "--batch_size",
            "1",
        ]
    )

    assert args.num_workers == 7
    assert args.train_set == train_set.resolve()
    assert args.test_set == test_set.resolve()
    assert args.batch_size == 1
    assert args.start_epoch == 0

    commands = []
    train_dataset = tmp_path / "temp.jsonl"
    train_dataset.write_text('{"instance_id":"task-0"}\n', encoding="utf-8")
    monkeypatch.setattr(cpu_main, "LOG_PATH", tmp_path / "log.out")
    monkeypatch.setattr(cpu_main, "_get_model_name", lambda _config_path: "model")
    monkeypatch.setattr(cpu_main, "get_success", lambda *_args: None)
    monkeypatch.setattr(
        cpu_main.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    cpu_main.run_sweagent(
        num_workers=args.num_workers,
        test_dataset=test_set,
        train_dataset=train_dataset,
        iteration_num=1,
        llm_api_key="key",
        llm_api_base="http://localhost:5001",
    )

    assert len(commands) == 2
    for command in commands:
        option_index = command.index("--num_workers")
        assert command[option_index + 1] == "7"
        assert "--parallel_instances" not in command
    test_command, train_command = commands
    test_subset_index = test_command.index("--instances.subset")
    train_subset_index = train_command.index("--instances.subset")
    assert test_command[test_subset_index + 1] == str(test_set)
    assert train_command[train_subset_index + 1] == str(train_dataset)


def test_cpu_cli_accepts_nonzero_start_epoch(tmp_path):
    train_set = tmp_path / "train.jsonl"
    test_set = tmp_path / "test.jsonl"
    train_set.write_text('{"instance_id":"train-task-0"}\n', encoding="utf-8")
    test_set.write_text('{"instance_id":"test-task-0"}\n', encoding="utf-8")

    args = cpu_main.parse_args(
        [
            "--num_workers",
            "1",
            "--train_set",
            str(train_set),
            "--test_set",
            str(test_set),
            "--batch_size",
            "1",
            "--start_epoch",
            "7",
        ]
    )

    assert args.start_epoch == 7


def test_cpu_cli_rejects_negative_start_epoch(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instance_id":"task-0"}\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        cpu_main.parse_args(
            [
                "--num_workers",
                "1",
                "--train_set",
                str(dataset),
                "--test_set",
                str(dataset),
                "--batch_size",
                "1",
                "--start_epoch",
                "-1",
            ]
        )


@pytest.mark.parametrize(
    ("epoch", "expected_ids"),
    [
        (0, ["task-0", "task-1", "task-2"]),
        (1, ["task-3", "task-4", "task-0"]),
        (2, ["task-1", "task-2", "task-3"]),
    ],
)
def test_writes_epoch_dataset_batch_with_wraparound(tmp_path, epoch, expected_ids):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"instance_id": f"task-{index}"}) for index in range(5)) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "temp.jsonl"

    result = cpu_main.write_epoch_dataset_batch(
        dataset,
        current_epoch=epoch,
        batch_size=3,
        output_path=output_path,
    )

    assert result == output_path
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["instance_id"] for record in records] == expected_ids


def test_epoch_batch_equal_boundaries_select_full_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instance_id":"task-0"}\n\n{"instance_id":"task-1"}\n', encoding="utf-8")

    output_path = cpu_main.write_epoch_dataset_batch(
        dataset,
        current_epoch=3,
        batch_size=2,
        output_path=tmp_path / "temp.jsonl",
    )

    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_epoch_batch_rejects_batch_larger_than_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instance_id":"task-0"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot exceed dataset size"):
        cpu_main.write_epoch_dataset_batch(
            dataset,
            current_epoch=0,
            batch_size=2,
            output_path=tmp_path / "temp.jsonl",
        )


def _write_trajectory(
    train_dir: Path,
    instance_id: str,
    sample_id: int,
    *,
    success: bool,
    input_ids: list[int] | None = None,
    output_ids: list[int] | None = None,
    probabilities: list[float] | None = None,
) -> None:
    sample_dir = train_dir / instance_id / str(sample_id)
    sample_dir.mkdir(parents=True)
    trajectory = []
    if input_ids is not None or output_ids is not None or probabilities is not None:
        trajectory.append(
            {
                "input_token_ids": input_ids,
                "output_token_ids": output_ids,
                "output_token_probabilities": probabilities,
            }
        )
    (sample_dir / f"{instance_id}.traj").write_text(
        json.dumps({"trajectory": trajectory, "info": {"success": success}}),
        encoding="utf-8",
    )


def test_collects_current_epoch_train_trajectories_in_dapo_format(tmp_path, monkeypatch):
    sweagent_dir = tmp_path / "sweagent"
    train_dir = sweagent_dir / "logs/model/3/train"
    test_dir = sweagent_dir / "logs/model/3/test"
    monkeypatch.setattr(cpu_main, "SWEAGENT_DIR", sweagent_dir)

    for instance_id in ["instance-b", "instance-a"]:
        _write_trajectory(
            train_dir,
            instance_id,
            0,
            success=True,
            input_ids=[10],
            output_ids=[20, 21],
            probabilities=[0.75, 0.25],
        )
        _write_trajectory(train_dir, instance_id, 1, success=False)
    _write_trajectory(
        test_dir,
        "test-only",
        0,
        success=True,
        input_ids=[99],
        output_ids=[100],
        probabilities=[1.0],
    )

    dapo_path = cpu_main.write_dapo_json(model="model", epoch=3, samples=2)

    assert dapo_path == train_dir / "dapo.json"
    assert json.loads(dapo_path.read_text(encoding="utf-8")) == [
        [[[[10], [[20, 0.75], [21, 0.25]], 1]], []],
        [[[[10], [[20, 0.75], [21, 0.25]], 1]], []],
    ]


def test_collection_inserts_placeholders_for_missing_samples(tmp_path, monkeypatch, capsys):
    sweagent_dir = tmp_path / "sweagent"
    train_dir = sweagent_dir / "logs/model/0/train"
    monkeypatch.setattr(cpu_main, "SWEAGENT_DIR", sweagent_dir)
    _write_trajectory(
        train_dir,
        "instance-a",
        0,
        success=True,
        input_ids=[10],
        output_ids=[20],
        probabilities=[0.75],
    )
    # Sample 1 has no directory. Sample 2 has a directory but no trajectory file.
    (train_dir / "instance-a/2").mkdir(parents=True)

    dapo_path = cpu_main.write_dapo_json(model="model", epoch=0, samples=3)

    assert json.loads(dapo_path.read_text(encoding="utf-8")) == [
        [[[[10], [[20, 0.75]], 1]], [], []]
    ]
    errors = capsys.readouterr().err
    assert "instance-a/1/instance-a.traj" in errors
    assert "instance-a/2/instance-a.traj" in errors


def test_collection_rejects_missing_token_metadata(tmp_path, monkeypatch):
    sweagent_dir = tmp_path / "sweagent"
    train_dir = sweagent_dir / "logs/model/0/train"
    monkeypatch.setattr(cpu_main, "SWEAGENT_DIR", sweagent_dir)
    _write_trajectory(
        train_dir,
        "instance-a",
        0,
        success=False,
        input_ids=[10],
        output_ids=[20],
        probabilities=None,
    )

    with pytest.raises(ValueError, match="output_token_probabilities must be a list"):
        cpu_main.write_dapo_json(model="model", epoch=0, samples=1)

    assert not (train_dir / "dapo.json").exists()


class _UploadState:
    def __init__(self, epoch: int, dapo_path: Path):
        self.epoch = epoch
        self.dapo_path = dapo_path
        self.upload_lock = threading.Lock()
        self.ready = False
        self.stop_requested = False
        self.normal_completion = False
        self.stop_event = threading.Event()

    def mark_trajectory_ready(self, epoch: int, _checksum: str) -> None:
        assert epoch == self.epoch
        self.ready = True

    def trajectory_is_ready(self) -> bool:
        return self.ready

    def request_stop(self, _reason: str, *, normal_completion: bool = False) -> None:
        self.stop_requested = True
        self.normal_completion = normal_completion
        self.stop_event.set()


class _RunningProcess:
    def poll(self):
        return None


class _Response:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int):
        return b""


def test_gpu_readiness_ack_retries_until_http_200(monkeypatch):
    responses = iter([_Response(503), _Response(200)])
    attempts = []
    epochs = []

    def fake_urlopen(request, *, timeout):
        attempts.append(timeout)
        epochs.append(request.get_header("X-dapo-epoch"))
        return next(responses)

    monkeypatch.setattr(gpu_server, "urlopen", fake_urlopen)

    assert gpu_server.send_ack_until_received(
        "http://127.0.0.1:8003",
        timeout=5,
        retry_interval=0,
        epoch=6,
        shutdown_event=threading.Event(),
        process=_RunningProcess(),
    )
    assert attempts == [5, 5]
    assert epochs == ["6", "6"]


def test_gpu_readiness_ack_default_retry_interval_is_one_minute():
    args, backend_args = gpu_server.parse_args(["--epoch", "0"])

    assert args.ack_retry_interval == 60.0
    assert args.listen_port == 8004
    assert backend_args == []


def test_gpu_server_succeeds_only_after_normal_epoch_completion():
    completed_state = SimpleNamespace(
        normal_completion=True,
        process=SimpleNamespace(returncode=-2),
    )
    failed_signal_state = SimpleNamespace(
        normal_completion=False,
        process=SimpleNamespace(returncode=-2),
    )
    failed_clean_exit_state = SimpleNamespace(
        normal_completion=False,
        process=SimpleNamespace(returncode=0),
    )
    failed_unknown_exit_state = SimpleNamespace(
        normal_completion=False,
        process=SimpleNamespace(returncode=None),
    )

    assert gpu_server.server_exit_code(completed_state) == 0
    assert gpu_server.server_exit_code(failed_signal_state) == -2
    assert gpu_server.server_exit_code(failed_clean_exit_state) == 1
    assert gpu_server.server_exit_code(failed_unknown_exit_state) == 1


def test_gpu_server_requires_both_current_upload_and_ack(tmp_path):
    state = gpu_server.RuntimeState(
        SimpleNamespace(returncode=0),
        stop_timeout=1,
        epoch=3,
        dapo_path=tmp_path / "dapo.json",
    )

    with pytest.raises(RuntimeError, match="before its DAPO data is received"):
        state.request_stop("ACK received", normal_completion=True)

    state.mark_trajectory_ready(epoch=3, checksum="a" * 64)
    assert not state.normal_completion
    assert gpu_server.server_exit_code(state) == 1


def test_cpu_completion_ack_retries_until_http_200_and_sends_epoch(monkeypatch):
    responses = iter([_Response(503), _Response(200)])
    epochs = []

    def fake_urlopen(request, *, timeout):
        assert timeout == 5
        epochs.append(request.get_header("X-dapo-epoch"))
        return next(responses)

    monkeypatch.setattr(cpu_main, "urlopen", fake_urlopen)

    assert cpu_main.send_ack_until_received(
        "http://127.0.0.1:8003",
        epoch=7,
        shutdown_event=threading.Event(),
        retry_interval=0,
    )
    assert epochs == ["7", "7"]
    assert cpu_main.DEFAULT_ACK_RETRY_INTERVAL == 60.0


def test_dapo_upload_retries_five_times_at_fifteen_second_intervals(
    tmp_path, monkeypatch
):
    attempts = []
    sleeps = []

    class UploadResponse:
        reason = "temporary"

        def __init__(self, status):
            self.status = status

        def read(self, _size):
            return b"temporary failure" if self.status == 503 else b"received"

    class FlakyConnection:
        def __init__(self, _host, _port, *, timeout):
            assert timeout == cpu_main.DEFAULT_TRAJECTORY_UPLOAD_TIMEOUT
            attempts.append(len(attempts) + 1)
            self.attempt = attempts[-1]

        def putrequest(self, _method, _path):
            if self.attempt <= 3:
                raise OSError("temporary tunnel failure")

        def putheader(self, *_args):
            pass

        def endheaders(self):
            pass

        def send(self, _chunk):
            pass

        def getresponse(self):
            return UploadResponse(503 if self.attempt == 4 else 201)

        def close(self):
            pass

    source = tmp_path / "dapo.json"
    source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cpu_main.http.client, "HTTPConnection", FlakyConnection)
    monkeypatch.setattr(cpu_main.time, "sleep", sleeps.append)

    cpu_main.upload_dapo_json("http://localhost:9001", source, epoch=3)

    assert cpu_main.DEFAULT_TRAJECTORY_UPLOAD_RETRIES == 5
    assert cpu_main.DEFAULT_TRAJECTORY_UPLOAD_RETRY_INTERVAL == 15.0
    assert attempts == [1, 2, 3, 4, 5]
    assert sleeps == [15.0, 15.0, 15.0, 15.0]


def _start_server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def test_cpu_readiness_ack_handler_records_epoch():
    ack_queue = queue.Queue()
    ack_server, ack_thread = _start_server(cpu_main.make_ack_handler(ack_queue))
    try:
        request = Request(
            f"http://127.0.0.1:{ack_server.server_port}",
            data=b"ACK",
            method="POST",
            headers={"X-DAPO-Epoch": "9"},
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 200
        assert ack_queue.get(timeout=2) == 9
    finally:
        ack_server.shutdown()
        ack_server.server_close()
        ack_thread.join(timeout=2)


def test_http_upload_atomically_replaces_dapo_and_gates_ack(tmp_path):
    destination = tmp_path / "verl/dapo.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("old data", encoding="utf-8")
    state = _UploadState(epoch=4, dapo_path=destination)
    upload_server, upload_thread = _start_server(gpu_server.make_trajectory_upload_handler(state))
    ack_server, ack_thread = _start_server(gpu_server.make_ack_handler(state))

    try:
        ack_url = f"http://127.0.0.1:{ack_server.server_port}"
        stale_request = Request(
            ack_url,
            data=b"ACK",
            method="POST",
            headers={"X-DAPO-Epoch": "3"},
        )
        with urlopen(stale_request, timeout=2) as response:
            assert response.status == 200
        assert not state.stop_requested

        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    ack_url,
                    data=b"ACK",
                    method="POST",
                    headers={"X-DAPO-Epoch": "4"},
                ),
                timeout=2,
            )
        assert error.value.code == 409
        assert not state.stop_requested

        source = tmp_path / "source.json"
        expected = [[[[[1, 2], [[3, 0.8], [4, 0.2]], 1]]]]
        source.write_text(json.dumps(expected), encoding="utf-8")
        cpu_main.upload_dapo_json(
            f"http://127.0.0.1:{upload_server.server_port}",
            source,
            epoch=4,
            timeout=2,
            retries=1,
        )

        assert json.loads(destination.read_text(encoding="utf-8")) == expected
        assert state.ready
        current_request = Request(
            ack_url,
            data=b"ACK",
            method="POST",
            headers={"X-DAPO-Epoch": "4"},
        )
        with urlopen(current_request, timeout=2) as response:
            assert response.status == 200
        assert state.stop_event.wait(timeout=2)
        assert state.stop_requested
        assert state.normal_completion
    finally:
        upload_server.shutdown()
        ack_server.shutdown()
        upload_server.server_close()
        ack_server.server_close()
        upload_thread.join(timeout=2)
        ack_thread.join(timeout=2)


def test_http_upload_rejects_wrong_epoch_without_overwriting(tmp_path):
    destination = tmp_path / "verl/dapo.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("old data", encoding="utf-8")
    state = _UploadState(epoch=5, dapo_path=destination)
    upload_server, upload_thread = _start_server(gpu_server.make_trajectory_upload_handler(state))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([[[[[1], [[2, 1.0]], 0]]]]), encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="HTTP 409"):
            cpu_main.upload_dapo_json(
                f"http://127.0.0.1:{upload_server.server_port}",
                source,
                epoch=4,
                timeout=2,
                retries=1,
            )
        assert destination.read_text(encoding="utf-8") == "old data"
        assert not state.ready
    finally:
        upload_server.shutdown()
        upload_server.server_close()
        upload_thread.join(timeout=2)
