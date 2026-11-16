"""Relocation must preserve dataset identity without depending on cwd or symlinks."""

import importlib.util
import json
from pathlib import Path

import pytest

from supermimic.utils.reproduction_paths import (
  data_root,
  legacy_motion_reference,
  portable_path,
  portable_tree,
  repo_path,
  resolve_motion_path,
  same_motion_source,
)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
  root = tmp_path / "new machine" / "checkout"
  root.mkdir(parents=True)
  elsewhere = tmp_path / "unrelated cwd"
  elsewhere.mkdir()
  monkeypatch.chdir(elsewhere)
  monkeypatch.delenv("SUPERMIMIC_DATA_DIR", raising=False)
  return root


def test_relative_paths_ignore_caller_directory(checkout):
  assert repo_path("reproduction/experiments/lafan1", root=checkout) == (
    checkout / "reproduction/experiments/lafan1"
  )
  assert data_root(root=checkout) == checkout / "reproduction/data"


@pytest.mark.parametrize("external", ["../shared motions", "absolute"])
def test_data_override_roundtrip(checkout, monkeypatch, external):
  storage = checkout.parent / "shared motions"
  setting = str(storage) if external == "absolute" else external
  monkeypatch.setenv("SUPERMIMIC_DATA_DIR", setting)
  reference = "reproduction/data/boneseed/packed_unfiltered"
  actual = resolve_motion_path(reference, root=checkout)
  assert actual == storage / "boneseed/packed_unfiltered"
  assert portable_path(actual, root=checkout) == reference


def test_globs_and_explicit_absolute_inputs(checkout, monkeypatch):
  monkeypatch.setenv("SUPERMIMIC_DATA_DIR", "../motions")
  assert resolve_motion_path("reproduction/data/lafan1/npz/*.npz", root=checkout) == (
    checkout.parent / "motions/lafan1/npz/*.npz"
  )
  explicit = checkout.parent / "another library/clip.npz"
  assert resolve_motion_path(explicit, root=checkout) == explicit


def test_symlink_model_name_survives_serialization(checkout):
  target = checkout / "archive/original_model"
  target.mkdir(parents=True)
  alias = checkout / "model"
  alias.symlink_to("archive/original_model", target_is_directory=True)
  assert portable_path(alias / "actor.pt", root=checkout) == "model/actor.pt"


def test_nested_config_is_portable_without_mutating_runtime(checkout, monkeypatch):
  monkeypatch.setenv("SUPERMIMIC_DATA_DIR", "../motions")
  physical = checkout.parent / "motions/lafan1/packed"
  config = {"args": {"out": checkout / "run", "motion_file": str(physical)}, "n": 42}
  saved = portable_tree(config, root=checkout)
  assert saved == {
    "args": {"out": "run", "motion_file": "reproduction/data/lafan1/packed"},
    "n": 42,
  }
  assert isinstance(config["args"]["out"], Path)
  assert config["args"]["motion_file"] == str(physical)


@pytest.mark.parametrize("layout", ["official_reproduction/data", "reproduction/data"])
def test_old_checkpoint_rebases_across_hosts(checkout, layout):
  old = checkout.parent / "removed server" / layout / "boneseed/packed_unfiltered"
  current = "reproduction/data/boneseed/packed_unfiltered"
  assert legacy_motion_reference(old, root=checkout) == current
  assert same_motion_source(old, current, root=checkout)
  assert not same_motion_source(
    old, "reproduction/data/boneseed/packed_filtered", root=checkout
  )
  assert not same_motion_source(
    old, "reproduction/data/other/packed_unfiltered", root=checkout
  )


def test_arbitrary_external_libraries_do_not_match_by_basename(checkout):
  a = checkout.parent / "library_a/packed"
  b = checkout.parent / "library_b/packed"
  assert not same_motion_source(a, b, root=checkout)


@pytest.mark.parametrize("dataset", ["packed_unfiltered", "packed_filtered"])
def test_real_resume_guard_accepts_relocation_but_rejects_dataset_change(
  checkout, monkeypatch, dataset
):
  # Exercise the actual runner's validation, stopping before any weights are read.
  script = Path(__file__).resolve().parents[1] / "scripts/training/train_campaign.py"
  spec = importlib.util.spec_from_file_location("campaign_portability_test", script)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  old = {
    "task_id": "same_task",
    "runner_cfg": {"dynamics_hidden_dims": (8, 4), "tasks": {"noise": (0.1, 0.2)}},
    "args": {
      "num_envs": 8,
      "motion_file": str(
        checkout.parent
        / "old_host/official_reproduction/data/boneseed/packed_unfiltered"
      ),
    },
  }
  current = json.loads(json.dumps(old))  # JSON represents tuples as lists.
  current["args"]["motion_file"] = f"reproduction/data/boneseed/{dataset}"
  monkeypatch.setattr(
    module.torch, "load", lambda *a, **kw: {"config": old, "world_model": None}
  )

  class ValidationPassed(Exception):
    pass

  class StopBeforeLoadingWeights:
    def load_bundle(self, value):
      raise ValidationPassed

  runner = module.CampaignRunner.__new__(module.CampaignRunner)
  runner.saved_config = current
  runner._wm = StopBeforeLoadingWeights()
  if dataset == "packed_unfiltered":
    with pytest.raises(ValidationPassed):
      runner.restore("unused.pt", None)
  else:
    with pytest.raises(ValueError, match="Resume dataset mismatch"):
      runner.restore("unused.pt", None)
