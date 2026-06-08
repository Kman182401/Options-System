"""ModelConfig: loads, validates, and expands the search grid deterministically."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from options_system.models.config import ModelConfig


def test_loads_default_and_counts_trials():
    cfg = ModelConfig.load()
    assert cfg.model_version
    assert cfg.target.timeout_handling in {"sign_return", "drop"}
    # default grid is 2 x 2 x 2 = 8 configs
    assert cfg.search.n_trials == 8
    assert len(cfg.search.configs()) == 8


def test_configs_are_unique_and_deterministic():
    cfg = ModelConfig.load()
    a = cfg.search.configs()
    b = cfg.search.configs()
    assert a == b  # deterministic order
    assert len({tuple(sorted(d.items())) for d in a}) == len(a)  # all distinct


def test_as_params_merges_and_int_casts():
    cfg = ModelConfig.load()
    params = cfg.lgbm.as_params({"max_depth": 5.0, "reg_lambda": 12.0})
    assert params["max_depth"] == 5 and isinstance(params["max_depth"], int)
    assert params["reg_lambda"] == 12.0
    # untouched base params survive
    assert params["learning_rate"] == cfg.lgbm.learning_rate


def test_rejects_unknown_timeout_handling(tmp_path):
    import yaml

    data = ModelConfig.load().to_dict()
    data["target"]["timeout_handling"] = "bogus"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        ModelConfig.load(p)


def test_rejects_unknown_selection_metric(tmp_path):
    import yaml

    data = ModelConfig.load().to_dict()
    data["search"]["selection_metric"] = "sortino"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        ModelConfig.load(p)
