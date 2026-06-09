"""Fold-local class weighting: computed from the training fold only, multiplied into
the sample weights, never leaking the global/full-sample class balance.
"""

from __future__ import annotations

import numpy as np

from options_system.microstructure_model.lgbm import (
    CLASSES,
    effective_sample_weight,
    fold_local_class_weights,
)


def test_balanced_property_equalises_effective_mass():
    # Imbalanced 3-class with sample weights; balanced weighting equalises cw*Σw per class.
    y = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 1])
    sw = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    cw = fold_local_class_weights(y, sw, CLASSES, use_sample_weight=True)
    masses = [cw[c] * sw[y == c].sum() for c in (-1, 0, 1)]
    assert np.allclose(masses, masses[0])  # all classes carry equal effective mass
    # the majority (timeout) class is down-weighted relative to the minorities
    assert cw[0] < cw[-1] and cw[0] < cw[1]


def test_weights_are_fold_local_not_global():
    # Two folds with DIFFERENT class balances must yield different weights, and neither
    # equals the weights computed on the full (global) sample — proving fold-locality.
    rng = np.random.default_rng(0)
    fold_a = np.array([-1] * 2 + [0] * 30 + [1] * 2)  # very timeout-heavy
    fold_b = np.array([-1] * 15 + [0] * 10 + [1] * 15)  # near-balanced
    full = np.concatenate([fold_a, fold_b])
    rng.shuffle(full)  # global balance differs from either fold
    wa = fold_local_class_weights(fold_a, None, CLASSES, use_sample_weight=False)
    wb = fold_local_class_weights(fold_b, None, CLASSES, use_sample_weight=False)
    wg = fold_local_class_weights(full, None, CLASSES, use_sample_weight=False)
    assert wa[0] != wb[0]  # different folds -> different class-0 weight
    assert wa[0] != wg[0] and wb[0] != wg[0]  # neither fold equals the global weighting


def test_effective_sample_weight_multiplies_class_into_sample():
    y = np.array([-1, 0, 1, 0])
    base = np.array([1.0, 2.0, 0.5, 4.0])
    cw = {-1: 3.0, 0: 0.5, 1: 3.0}
    eff = effective_sample_weight(y, base, cw)
    assert np.allclose(eff, [1.0 * 3.0, 2.0 * 0.5, 0.5 * 3.0, 4.0 * 0.5])


def test_missing_class_in_fold_does_not_raise():
    # A fold missing the +1 class: balanced over present classes, absent class -> 1.0.
    y = np.array([-1, -1, 0, 0, 0])
    cw = fold_local_class_weights(y, None, CLASSES, use_sample_weight=False)
    assert set(cw) == set(CLASSES)
    assert cw[1] == 1.0  # absent class defaults to neutral weight
    assert cw[0] < cw[-1]  # present-class balancing still down-weights the majority


def test_class_weight_uses_only_training_rows():
    # Construct full y where the global majority is class +1, but the TRAIN fold's
    # majority is class 0. The fold-local weight must reflect the fold, not the global.
    full = np.array([1] * 50 + [-1] * 5 + [0] * 5)
    train_idx = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])  # the -1 and 0 rows only
    cw_train = fold_local_class_weights(full[train_idx], None, CLASSES, use_sample_weight=False)
    # within the train fold, classes -1 and 0 are balanced (5 each) -> equal weights;
    # class +1 is absent from the fold -> neutral 1.0 (NOT the tiny global +1 weight).
    assert cw_train[-1] == cw_train[0]
    assert cw_train[1] == 1.0
