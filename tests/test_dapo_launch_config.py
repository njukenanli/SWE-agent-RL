from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DAPO = ROOT / "gpu/verl/examples/sft/rft/dapo.py"


def _load_torchrun_command():
    module = ast.parse(DAPO.read_text(encoding="utf-8"), filename=str(DAPO))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_torchrun_command"
    )
    namespace = {"__file__": str(DAPO), "argparse": argparse, "Path": Path}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(DAPO), "exec"),
        namespace,
    )
    return namespace["_torchrun_command"]


def test_qwen35_dapo_uses_bshd_without_dynamic_batching(tmp_path):
    args = SimpleNamespace(
        nnodes=1,
        nproc=1,
        node_rank=0,
        master_addr="localhost",
        master_port="29600",
        train_parquet=tmp_path / "dapo.parquet",
        micro_batch_size_per_gpu=1,
        max_token_len_per_gpu=60_000,
        max_total_length=60_000,
        num_workers=0,
        model_path="Qwen/Qwen3.5-4B",
        lr="1e-6",
        min_lr="1e-6",
        weight_decay="0.1",
        tp=1,
        pp=1,
        cp=1,
        save_path=tmp_path / "checkpoint",
        project_name="offline-dapo",
        experiment_name="qwen35-test",
        resume_mode="disable",
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
    )

    command = _load_torchrun_command()(args, global_batch_size=2)

    assert "model.use_remove_padding=False" in command
    assert "model.use_remove_padding=True" not in command
    assert "data.use_dynamic_bsz=False" in command
    assert "engine.vanilla_mbridge=True" in command
    assert "engine.vanilla_mbridge=False" not in command
    assert "engine.use_distributed_optimizer=True" in command
    assert "engine.use_megatron_fsdp=False" in command
    assert "engine.use_megatron_fsdp=True" not in command
