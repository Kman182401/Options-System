# models/

**Trains, versions, and serves the signal model — entirely offline.** The model
is **LightGBM** (gradient-boosted trees: fast on CPU, interpretable). This
module owns: training runs over the feature set, a simple **model registry**
(versioned artifacts on disk under `models/` with their metadata and metrics),
and the **champion–challenger** gate — a new ("challenger") model only replaces
the live ("champion") model after it beats it on out-of-sample, walk-forward
evaluation with realistic costs. Training is part of the **offline learning
loop**, never the live loop; the live engine merely *loads* an already-approved
champion artifact and runs inference. Overfitting is the enemy here: no model
is promoted on in-sample results.
