"""Labeling layer — the supervised-learning *target* (not trading logic).

Triple-barrier labels (López de Prado, *Advances in Financial Machine Learning*
ch. 3) for MES/MNQ: from each event ``t0``, does the price hit ``+pt·σ`` before
``−sl·σ`` within a max hold, or does the vertical (time) barrier expire first?
The label is the answer (+1 / −1 / 0). Volatility-scaled barriers encode the
system's defined-risk, intraday→3-day style as a learnable target.

Modules:

* :mod:`~options_system.labeling.config` — declarative, versioned config.
* :mod:`~options_system.labeling.events` — causal σ_t + CUSUM/grid event sampler.
* :mod:`~options_system.labeling.triple_barrier` — first-touch label generator.
* :mod:`~options_system.labeling.weights` — concurrency → average-uniqueness weights.
* :mod:`~options_system.labeling.build` — versioned label-table writer + retrieval.

**Leakage contract.** A label is *allowed* to look forward (that is what a target
is); a feature is not. The hard rule, enforced here: labels never feed back into
the feature set, and every label records its resolution time ``t1`` so the next
phase (validation) can purge/embargo overlapping samples. Barrier touches are
computed in return/log space on the real tradeable instrument, so they are
invariant to back-adjustment. See ``docs/LABELING.md``.
"""

from __future__ import annotations
