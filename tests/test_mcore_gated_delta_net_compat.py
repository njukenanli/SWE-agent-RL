from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MCORE_PATCH = ROOT / "gpu/verl/verl/models/mcore/patch.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("verl_mcore_patch_test", MCORE_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_gated_delta_net_module(monkeypatch, gated_delta_net_class):
    megatron = ModuleType("megatron")
    megatron_core = ModuleType("megatron.core")
    megatron_ssm = ModuleType("megatron.core.ssm")
    gated_delta_net = ModuleType("megatron.core.ssm.gated_delta_net")
    gated_delta_net.GatedDeltaNet = gated_delta_net_class
    megatron.core = megatron_core
    megatron_core.ssm = megatron_ssm
    megatron_ssm.gated_delta_net = gated_delta_net

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(sys.modules, "megatron.core.ssm", megatron_ssm)
    monkeypatch.setitem(sys.modules, "megatron.core.ssm.gated_delta_net", gated_delta_net)


def test_old_gated_delta_net_accepts_and_ignores_cp_comm_type(monkeypatch):
    class OldGatedDeltaNet:
        def __init__(self, config, *, name=None):
            self.config = config
            self.name = name

    _install_fake_gated_delta_net_module(monkeypatch, OldGatedDeltaNet)
    patch = _load_patch_module()

    patch._patch_gated_delta_net_cp_comm_type()
    patched_init = OldGatedDeltaNet.__init__
    layer = OldGatedDeltaNet("config", name="layer", cp_comm_type="a2a")

    assert layer.config == "config"
    assert layer.name == "layer"
    assert getattr(patched_init, "_verl_cp_comm_type_compat") is True

    patch._patch_gated_delta_net_cp_comm_type()
    assert OldGatedDeltaNet.__init__ is patched_init


def test_new_gated_delta_net_constructor_is_not_modified(monkeypatch):
    class NewGatedDeltaNet:
        def __init__(self, config, *, cp_comm_type=None):
            self.config = config
            self.cp_comm_type = cp_comm_type

    _install_fake_gated_delta_net_module(monkeypatch, NewGatedDeltaNet)
    patch = _load_patch_module()
    original_init = NewGatedDeltaNet.__init__

    patch._patch_gated_delta_net_cp_comm_type()
    layer = NewGatedDeltaNet("config", cp_comm_type="a2a")

    assert NewGatedDeltaNet.__init__ is original_init
    assert layer.cp_comm_type == "a2a"
