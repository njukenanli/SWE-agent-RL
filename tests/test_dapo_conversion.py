from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
DAPO_PATH = ROOT / "gpu/verl/examples/sft/rft/dapo.py"


@pytest.fixture(scope="module")
def dapo() -> ModuleType:
    """Load the conversion code without requiring the GPU PyTorch stack."""

    class Dataset:
        pass

    class DistributedSampler:
        pass

    class StatefulDataLoader:
        pass

    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")
    torch_data.Dataset = Dataset
    torch_data.DistributedSampler = DistributedSampler
    torch.utils = torch_utils
    torch_utils.data = torch_data

    torchdata = types.ModuleType("torchdata")
    torchdata_stateful = types.ModuleType("torchdata.stateful_dataloader")
    torchdata_stateful.StatefulDataLoader = StatefulDataLoader

    replacements = {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_data,
        "torchdata": torchdata,
        "torchdata.stateful_dataloader": torchdata_stateful,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location("dapo_conversion_under_test", DAPO_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _step(reward: int, *, seed: int) -> list[object]:
    return [[seed, seed + 1], [[seed + 2, 0.75], [seed + 3, 0.25]], reward]


def _convert(dapo: ModuleType, tmp_path: Path, groups: list[object], **overrides):
    json_path = tmp_path / "dapo.json"
    parquet_path = tmp_path / "dapo.parquet"
    json_path.write_text(json.dumps(groups), encoding="utf-8")
    kwargs = {
        "num_groups": 0,
        "expected_trajs_per_group": 0,
        "max_input_length": 100,
        "max_total_length": 100,
        "max_steps": 10,
        "step_penalty_threshold": 10,
        "min_probability": 1e-12,
        "std_epsilon": 1e-6,
        "std_ddof": 0,
        "zero_adv_epsilon": 0.0,
        "divisibility": 1,
    }
    kwargs.update(overrides)
    stats = dapo.convert_json_to_parquet(json_path, parquet_path, **kwargs)
    return stats, dapo.pd.read_parquet(parquet_path)


def test_normalizes_once_per_trajectory_and_copies_advantage_to_output_tokens(
    dapo: ModuleType, tmp_path: Path
) -> None:
    rewards = [1, 0, 1, 0, 0, 0, 1, 0]
    step_counts = [1, 3, 2, 4, 1, 5, 2, 3]
    group = [
        [_step(reward, seed=100 * trajectory_index + step_index) for step_index in range(step_count)]
        for trajectory_index, (reward, step_count) in enumerate(zip(rewards, step_counts, strict=True))
    ]

    stats, frame = _convert(dapo, tmp_path, [group])

    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    assert stats["group_size"] == 8
    assert stats["real_rows"] == sum(step_counts)
    for trajectory_index, reward in enumerate(rewards):
        expected_advantage = (reward - mean) / (std + 1e-6)
        trajectory_rows = frame[frame["traj_idx"] == trajectory_index]
        assert len(trajectory_rows) == step_counts[trajectory_index]
        assert trajectory_rows["advantage"].tolist() == pytest.approx(
            [expected_advantage] * step_counts[trajectory_index]
        )
        for row in trajectory_rows.itertuples():
            token_advantages = list(row.advantages)
            assert token_advantages[:2] == [0.0, 0.0]
            assert token_advantages[2:] == pytest.approx([expected_advantage, expected_advantage])


def test_normalizes_penalized_trajectory_rewards(dapo: ModuleType, tmp_path: Path) -> None:
    raw_rewards = [1, 0, 1, 0, 0, 0, 1, 0]
    step_counts = [1, 2, 3, 4, 1, 2, 3, 4]
    group = [
        [_step(reward, seed=100 * trajectory_index + step_index) for step_index in range(step_count)]
        for trajectory_index, (reward, step_count) in enumerate(zip(raw_rewards, step_counts, strict=True))
    ]

    _stats, frame = _convert(
        dapo,
        tmp_path,
        [group],
        max_steps=4,
        step_penalty_threshold=2,
    )

    final_rewards = [
        reward + dapo._step_penalty(step_count, threshold=2, max_step=4)
        for reward, step_count in zip(raw_rewards, step_counts, strict=True)
    ]
    mean = statistics.fmean(final_rewards)
    std = statistics.pstdev(final_rewards)
    for trajectory_index, final_reward in enumerate(final_rewards):
        expected_advantage = (final_reward - mean) / (std + 1e-6)
        trajectory_rows = frame[frame["traj_idx"] == trajectory_index]
        if expected_advantage == 0.0:
            assert trajectory_rows.empty
            continue
        assert trajectory_rows["effective_reward"].tolist() == pytest.approx(
            [final_reward] * step_counts[trajectory_index]
        )
        assert trajectory_rows["advantage"].tolist() == pytest.approx(
            [expected_advantage] * step_counts[trajectory_index]
        )


def test_discards_uniform_group_and_counts_empty_trajectories_in_normalization(
    dapo: ModuleType, tmp_path: Path
) -> None:
    uniform_group = [[_step(0, seed=trajectory_index * 10)] for trajectory_index in range(8)]
    mixed_group = [[_step(1, seed=1000)]] + [[] for _ in range(7)]

    stats, frame = _convert(dapo, tmp_path, [uniform_group, mixed_group])

    expected_advantage = (1.0 - 0.125) / (math.sqrt(0.109375) + 1e-6)
    assert stats["groups_with_zero_advantage"] == 1
    assert stats["discard_zero_advantage"] == 8
    assert frame["group_idx"].tolist() == [1]
    assert frame["traj_idx"].tolist() == [0]
    assert frame["advantage"].tolist() == pytest.approx([expected_advantage])


def test_detects_non_eight_group_size_from_dapo_json(dapo: ModuleType, tmp_path: Path) -> None:
    rewards = [1, 0, 1]
    group = [[_step(reward, seed=trajectory_index * 10)] for trajectory_index, reward in enumerate(rewards)]

    stats, frame = _convert(dapo, tmp_path, [group])

    assert dapo._arg_parser().parse_args([]).expected_trajs_per_group == 0
    assert stats["group_size"] == 3
    assert stats["trajectories"] == 3
    assert sorted(frame["traj_idx"].tolist()) == [0, 1, 2]


def test_auto_detected_group_size_rejects_incomplete_groups(dapo: ModuleType, tmp_path: Path) -> None:
    complete_group = [[_step(reward, seed=index * 10)] for index, reward in enumerate([1, 0, 1])]
    incomplete_group = [[_step(reward, seed=100 + index * 10)] for index, reward in enumerate([1, 0])]

    with pytest.raises(ValueError, match="group 1: expected 3 trajectories, got 2"):
        _convert(dapo, tmp_path, [complete_group, incomplete_group])
