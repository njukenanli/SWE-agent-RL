from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GPU_MAIN = ROOT / "gpu/main.sh"


def _write_fake_python(fake_bin: Path) -> None:
    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "  printf '%s\\n' \"${FAKE_CUDA_DEVICE_COUNT:-1}\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"-\" ]]; then\n"
        "  exit \"${FAKE_DEPENDENCY_CHECK_EXIT:-0}\"\n"
        "fi\n"
        "printf '%s\\n' \"$*\" > \"$CAPTURE_PATH\"\n"
        "exit 23\n",
        encoding="utf-8",
    )
    python.chmod(0o755)


def _write_hf_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")


@pytest.mark.parametrize(
    ("extra_args", "expected_model", "expected_epoch"),
    [
        ([], "Qwen/Qwen3.5-4B", "0"),
        (["--start_epoch", "3"], "model/epoch_2", "3"),
    ],
)
def test_gpu_main_starts_with_expected_model(
    tmp_path, extra_args, expected_model, expected_epoch
):
    data_dir = tmp_path / "data"
    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "python-args.txt"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)
    if extra_args:
        _write_hf_model(data_dir / "model/epoch_2")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "4",
            "--data-dir",
            str(data_dir),
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    command = capture_path.read_text(encoding="utf-8").strip().split()
    assert command[:2] == ["vllm/server.py", "--model"]
    if extra_args:
        assert command[2].endswith(expected_model)
    else:
        assert command[2] == expected_model
    assert command[3:] == [
        "--epoch",
        expected_epoch,
        "--",
        "--tensor-parallel-size",
        "1",
    ]


@pytest.mark.parametrize(
    ("extra_args", "expected_tensor_parallel"),
    [
        ([], "8"),
        (["--vllm-tensor-parallel", "4"], "4"),
    ],
)
def test_gpu_main_configures_vllm_tensor_parallelism(
    tmp_path, extra_args, expected_tensor_parallel
):
    data_dir = tmp_path / "data"
    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "python-args.txt"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["FAKE_CUDA_DEVICE_COUNT"] = "8"
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(data_dir),
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    command = capture_path.read_text(encoding="utf-8").strip().split()
    tensor_parallel_index = command.index("--tensor-parallel-size")
    assert command[tensor_parallel_index + 1] == expected_tensor_parallel
    assert f"vLLM will use tensor parallelism {expected_tensor_parallel}" in result.stdout


def test_gpu_main_serves_exported_hf_checkpoint_in_next_epoch(tmp_path):
    data_dir = tmp_path / "data"
    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "vllm-args.txt"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "  printf '1\\n'\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"vllm/server.py\" ]]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$CAPTURE_PATH\"\n"
        "  if [[ $(wc -l < \"$CAPTURE_PATH\") -ge 2 ]]; then\n"
        "    exit 23\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"examples/sft/rft/dapo.py\" ]]; then\n"
        "  shift\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    if [[ \"$1\" == \"--save-path\" ]]; then\n"
        "      save_path=\"$2\"\n"
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "  hf_path=\"$save_path/global_step_1/model/huggingface\"\n"
        "  mkdir -p \"$hf_path\"\n"
        "  printf '{}\\n' > \"$hf_path/config.json\"\n"
        "  printf '{}\\n' > \"$hf_path/tokenizer_config.json\"\n"
        "  printf 'weights' > \"$hf_path/model.safetensors\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "2",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    commands = [line.split() for line in capture_path.read_text(encoding="utf-8").splitlines()]
    assert commands[0][0:3] == ["vllm/server.py", "--model", "Qwen/Qwen3.5-4B"]
    assert commands[1][0:2] == ["vllm/server.py", "--model"]
    assert Path(commands[1][2]).resolve() == (
        data_dir / "checkpoints/epoch_0/global_step_1/model/huggingface"
    ).resolve()
    assert commands[1][3:5] == ["--epoch", "1"]
    assert (data_dir / "model/epoch_0").is_symlink()


def test_gpu_main_aborts_when_current_epoch_has_complete_model(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = data_dir / "checkpoints/epoch_0"
    model_path = data_dir / "model/epoch_0"
    _write_hf_model(checkpoint_path / "global_step_1/model/huggingface")
    model_path.parent.mkdir(parents=True)
    model_path.symlink_to("../checkpoints/epoch_0/global_step_1/model/huggingface")

    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "python-args.txt"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        f"Model path already exists: {model_path}. If you want to skip this epoch, "
        f"check the model checkpoint under {model_path} is complete and then increase "
        "start_epoch arg; if you want to redo this epoch, manually delete model "
        f"checkpoints at {model_path} and all checkpoints of epochs after it."
    ) in result.stderr
    assert checkpoint_path.is_dir()
    assert model_path.is_symlink()
    assert model_path.is_dir()
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("create_checkpoint", "model_state"),
    [
        (True, "missing"),
        (False, "complete"),
        (True, "incomplete"),
        (True, "broken_symlink"),
    ],
)
def test_gpu_main_removes_incomplete_current_epoch_and_retries(
    tmp_path, create_checkpoint, model_state
):
    data_dir = tmp_path / "data"
    checkpoint_path = data_dir / "checkpoints/epoch_0"
    model_path = data_dir / "model/epoch_0"

    if create_checkpoint:
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "partial").write_text("partial\n", encoding="utf-8")
    if model_state == "complete":
        _write_hf_model(model_path)
    elif model_state == "incomplete":
        model_path.mkdir(parents=True)
        (model_path / "partial").write_text("partial\n", encoding="utf-8")
    elif model_state == "broken_symlink":
        model_path.parent.mkdir(parents=True)
        model_path.symlink_to("missing-hf-export")

    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "python-args.txt"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    assert not checkpoint_path.exists()
    assert not checkpoint_path.is_symlink()
    assert not model_path.exists()
    assert not model_path.is_symlink()
    command = capture_path.read_text(encoding="utf-8").strip().split()
    assert command[:3] == ["vllm/server.py", "--model", "Qwen/Qwen3.5-4B"]


@pytest.mark.parametrize(
    ("gpu_count", "extra_args", "expected_context_parallel"),
    [
        ("1", [], "1"),
        ("4", [], "1"),
        ("8", ["--context-parallel", "2"], "2"),
    ],
)
def test_gpu_main_uses_visible_gpu_count_and_context_parallelism(
    tmp_path, gpu_count, extra_args, expected_context_parallel
):
    data_dir = tmp_path / "data"
    fake_bin = tmp_path / "bin"
    capture_path = tmp_path / "python-args.txt"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "  printf '%s\\n' \"$FAKE_CUDA_DEVICE_COUNT\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"-\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"vllm/server.py\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$*\" > \"$CAPTURE_PATH\"\n"
        "exit 23\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["FAKE_CUDA_DEVICE_COUNT"] = gpu_count
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(data_dir),
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    command = capture_path.read_text(encoding="utf-8").strip().split()
    assert command[:2] == ["examples/sft/rft/dapo.py", "--dapo-json"]
    nproc_index = command.index("--nproc")
    assert command[nproc_index + 1] == gpu_count
    cp_index = command.index("--cp")
    assert command[cp_index + 1] == expected_context_parallel
    assert f"Detected {gpu_count} visible CUDA GPU(s)" in result.stdout


@pytest.mark.parametrize("context_parallel", ["0", "not-an-integer"])
def test_gpu_main_rejects_invalid_context_parallel(tmp_path, context_parallel):
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
            "--context-parallel",
            context_parallel,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--context-parallel must be a positive integer" in result.stderr


def test_gpu_main_rejects_context_parallel_not_dividing_gpu_count(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    capture_path = tmp_path / "python-args.txt"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["FAKE_CUDA_DEVICE_COUNT"] = "8"
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
            "--context-parallel",
            "3",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Detected GPU count (8) must be divisible by --context-parallel (3)" in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize("tensor_parallel", ["0", "not-an-integer"])
def test_gpu_main_rejects_invalid_vllm_tensor_parallel(tmp_path, tensor_parallel):
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
            "--vllm-tensor-parallel",
            tensor_parallel,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--vllm-tensor-parallel must be a positive integer" in result.stderr


def test_gpu_main_rejects_vllm_tensor_parallel_above_visible_gpu_count(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    capture_path = tmp_path / "python-args.txt"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["FAKE_CUDA_DEVICE_COUNT"] = "8"
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
            "--vllm-tensor-parallel",
            "9",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot exceed the visible GPU count (8)" in result.stderr
    assert not capture_path.exists()


def test_gpu_main_rejects_no_visible_gpus(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(tmp_path / "python-args.txt")
    env["FAKE_CUDA_DEVICE_COUNT"] = "0"
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Expected at least one visible CUDA GPU, detected: 0" in result.stderr
    assert not (tmp_path / "python-args.txt").exists()


def test_gpu_main_stops_when_dependency_validation_fails(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_python(fake_bin)

    capture_path = tmp_path / "python-args.txt"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["FAKE_DEPENDENCY_CHECK_EXIT"] = "7"
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "1",
            "--data-dir",
            str(tmp_path / "data"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    assert not capture_path.exists()


def test_gpu_main_rejects_start_epoch_at_or_after_target(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(GPU_MAIN),
            "--epoch",
            "3",
            "--data-dir",
            str(tmp_path / "data"),
            "--start_epoch",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--start_epoch must be less than --epoch" in result.stderr
