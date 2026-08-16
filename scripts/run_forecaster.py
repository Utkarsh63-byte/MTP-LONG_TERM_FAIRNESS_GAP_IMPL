"""Train and score the forecaster (paper Sec. 4.2 + Gap 1a congestion head).

Run:  .venv/bin/python -m scripts.run_forecaster
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from ltf.config import Config
from ltf.data import demand as demand_mod
from ltf.data import graph as graph_mod
from ltf.data import traveltime as tt_mod
from ltf.predict import Forecaster
from ltf.utils import LOG, output_path


def main() -> int:
    cfg = Config()
    graph = graph_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=graph.n_nodes)

    # train-only congestion (features/targets) and test-week congestion (scoring)
    tt_train = tt_mod.build(cfg, graph, date_lt=cfg.protocol.test_start,
                            name="traveltime_train.npz")
    trips = None
    from ltf.data.prepare import load_trips
    trips = load_trips(columns=["o", "d", "dur_min", "km", "is_weekend",
                                "hour", "date"])
    tt_test = tt_mod.build(cfg, graph,
                           trips=trips[trips.date >= cfg.protocol.test_start],
                           name="traveltime_test.npz")

    fc = Forecaster(cfg)
    res = fc.fit(cfg, dem, graph, tt_train, tt_test)

    print("\n" + "=" * 78)
    print("FORECASTER RESULTS")
    print("=" * 78)
    print(f"  device                 {res.device}")
    print(f"  train / test samples   {res.n_train:,} / {res.n_test:,}")
    print(f"  epochs run             {res.epochs_run}")
    print(f"  demand MSE (log space) {res.demand_mse:.5f}")
    print(f"  demand MSE (counts)    {res.demand_mse_counts:.3f}")
    print(f"  seasonal-naive MSE     {res.baseline_mse_counts:.3f}")
    print(f"  improvement over naive {100*(1-res.demand_mse_counts/res.baseline_mse_counts):.1f}%")
    print(f"  congestion MSE / MAE   {res.congestion_mse:.5f} / {res.congestion_mae:.4f}")
    print(f"\n  paper reports request-prediction MSE 94.69 (Sec. 5.4). Ours is on")
    print(f"  per-OD-pair hourly counts at 87 nodes; the paper never states its")
    print(f"  node count or aggregation level, so the two are not directly")
    print(f"  comparable. The seasonal-naive comparison is the meaningful check,")
    print(f"  and the congestion head is what Gap 1a actually needs.")

    with open(output_path("forecaster_results.json"), "w") as f:
        json.dump(asdict(res), f, indent=2)
    LOG.info("wrote outputs/forecaster_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
