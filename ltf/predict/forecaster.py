"""Request forecaster (paper Sec. 4.2) with an added congestion head (Gap 1a).

Paper's module: a 3-layer MLP over lagged per-OD-pair hourly counts, trained on
one month, forecasting the next 7 days; predicted requests then enter the MDP's
action space. Reported MSE 94.69.

Two things this implementation is careful about.

1. No leakage in the 7-day forecast. A naive recursive forecast would need the
   test week's own counts as lags. Instead the features use *seasonal* lags -- the
   same hour one and two weeks earlier -- which for every hour of the test week
   fall strictly before it. Test hours are 576..743 and h-168 spans 408..575, so
   the forecast origin is respected without any recursive drift.

2. The congestion head is trained on train-only congestion. Gap 1a's utility
   needs c_t for *future* slots. Reading it from the full-month tensor would leak,
   so `traveltime.build(date_lt=test_start)` supplies the targets and the test
   week's realised congestion is used only for scoring. The environment keeps
   using the full-month tensor as its physics; the policy only ever sees
   predictions. Sharing the trunk between the two heads is what the gap document
   asks for -- reuse the existing demand-prediction module rather than bolt on a
   second model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..utils import LOG, cache_path, resolve_device, timed

# Only lags of >= 168 hours are usable. Anything shorter (e.g. lag24) would sit
# inside the test week and leak, since the whole week is forecast from one
# origin. A 2-week lag was tried and dropped: requiring h >= 336 left just 72
# training hours, which collapsed the day-of-week features to near-zero variance
# and blew up the input normalisation.
FEATURE_NAMES = [
    "lag168", "lag167", "lag169", "lag170",       # window around same hour last week
    "pair_mean", "orig_lag168", "dest_lag168",    # level information
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "orig_volume", "dest_volume", "dist_km", "tau_bar_min",
]
MIN_LAG = 171          # earliest hour with every lag available
SD_FLOOR = 1e-2        # guards against a near-constant feature exploding
CLIP = 10.0            # clamp on normalised features
N_FEATURES = len(FEATURE_NAMES)
CKPT = "forecaster.pt"


@dataclass
class ForecastResult:
    demand_mse: float
    demand_mae: float
    demand_mse_counts: float
    congestion_mse: float
    congestion_mae: float
    baseline_mse_counts: float
    n_train: int
    n_test: int
    device: str
    epochs_run: int


def _calendar(hours: np.ndarray) -> tuple[np.ndarray, ...]:
    hod = hours % 24
    dow = ((hours // 24) + 1) % 7          # 2016-03-01 was a Tuesday
    return (np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(np.float32))


def build_dataset(cfg: Config, demand, graph, tt_train, pairs: np.ndarray,
                  hours: np.ndarray, pair_mean: np.ndarray,
                  orig_volume: np.ndarray, dest_volume: np.ndarray):
    """Assemble (features, demand target, congestion target) for pair x hour."""
    od = demand.od
    org = demand.org
    o = pairs[:, 0]
    d = pairs[:, 1]
    P, H = len(pairs), len(hours)
    oo = np.repeat(o, H)
    dd = np.repeat(d, H)
    hh = np.tile(hours, P)

    def lag(k):
        idx = np.clip(hh - k, 0, od.shape[2] - 1)
        return np.log1p(od[oo, dd, idx].astype(np.float32))

    hs, hc, ds, dc, wk = _calendar(hh)
    prof = (wk.astype(np.int64) * 24 + (hh % 24)).astype(np.int64)

    X = np.stack([
        lag(168), lag(167), lag(169), lag(170),
        np.log1p(np.repeat(pair_mean, H)),
        np.log1p(org[oo, np.clip(hh - 168, 0, org.shape[1] - 1)].astype(np.float32)),
        np.log1p(org[dd, np.clip(hh - 168, 0, org.shape[1] - 1)].astype(np.float32)),
        hs, hc, ds, dc, wk,
        np.log1p(np.repeat(orig_volume, H)),
        np.log1p(np.repeat(dest_volume, H)),
        graph.dist[oo, dd].astype(np.float32),
        tt_train.tau_bar[oo, dd].astype(np.float32),
    ], axis=1).astype(np.float32)

    y_dem = np.log1p(od[oo, dd, hh].astype(np.float32))
    y_con = tt_train.c[oo, dd, prof].astype(np.float32)
    return X, y_dem, y_con, (oo, dd, hh, prof)


class Forecaster:
    """3-layer MLP, two heads, device-agnostic (CPU / MPS / CUDA)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.predict.device)
        self.model = None
        self.mu = None
        self.sd = None

    # ------------------------------------------------------------------
    def _build_model(self):
        import torch.nn as nn

        p = self.cfg.predict
        layers = []
        prev = N_FEATURES
        for h in p.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p.dropout)]
            prev = h
        trunk = nn.Sequential(*layers)

        class TwoHead(nn.Module):
            def __init__(self, trunk, width, with_congestion):
                super().__init__()
                self.trunk = trunk
                self.demand = nn.Linear(width, 1)
                self.congestion = nn.Linear(width, 1) if with_congestion else None

            def forward(self, x):
                z = self.trunk(x)
                dem = self.demand(z).squeeze(-1)
                con = (self.congestion(z).squeeze(-1)
                       if self.congestion is not None else None)
                return dem, con

        return TwoHead(trunk, prev, p.congestion_head)

    # ------------------------------------------------------------------
    def fit(self, cfg: Config, demand, graph, tt_train, tt_test) -> ForecastResult:
        import torch

        torch.manual_seed(cfg.predict.seed)
        K = graph.n_nodes
        # pairs worth forecasting: those the month actually saw
        tot = demand.od.sum(axis=2)
        pairs = np.argwhere(tot >= 30)
        pair_mean = tot[pairs[:, 0], pairs[:, 1]] / demand.n_hours
        vol = demand.org.sum(axis=1)
        LOG.info("forecaster: %d OD pairs x hourly steps, device=%s",
                 len(pairs), self.device)

        h_all = np.arange(MIN_LAG, demand.n_hours)
        from ..sim.timeline import EPOCH
        import pandas as pd
        test_h0 = int((pd.Timestamp(cfg.protocol.test_start) - EPOCH)
                      / pd.Timedelta(hours=1))
        # validation is the 48 hours immediately before the forecast origin;
        # everything earlier trains. Using the whole sim-train week for
        # validation would leave too little history to learn a weekly cycle.
        val_h0 = test_h0 - 48
        h_train = h_all[h_all < val_h0]
        h_val = h_all[(h_all >= val_h0) & (h_all < test_h0)]
        h_test = h_all[h_all >= test_h0]
        LOG.info("  hours: train %d, val %d, test %d (test starts at hour %d)",
                 len(h_train), len(h_val), len(h_test), test_h0)

        def make(hs):
            return build_dataset(cfg, demand, graph, tt_train, pairs, hs,
                                 pair_mean, vol[pairs[:, 0]], vol[pairs[:, 1]])

        Xtr, ytr_d, ytr_c, _ = make(h_train)
        Xva, yva_d, yva_c, _ = make(h_val)
        Xte, yte_d, yte_c, idx_te = make(h_test)

        self.mu = Xtr.mean(0)
        self.sd = np.maximum(Xtr.std(0), SD_FLOOR)
        norm = lambda X: np.clip((X - self.mu) / self.sd, -CLIP, CLIP)

        dev = torch.device(self.device)
        self.model = self._build_model().to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.predict.lr,
                                weight_decay=cfg.predict.weight_decay)
        lossf = torch.nn.MSELoss()

        Xtr_t = torch.tensor(norm(Xtr), device=dev)
        ytr_d_t = torch.tensor(ytr_d, device=dev)
        ytr_c_t = torch.tensor(ytr_c, device=dev)
        Xva_t = torch.tensor(norm(Xva), device=dev)
        yva_d_t = torch.tensor(yva_d, device=dev)

        n = len(Xtr_t)
        bs = max(cfg.predict.batch_size, 4096)
        best, best_state, bad, epochs_run = np.inf, None, 0, 0
        with timed(f"training forecaster on {self.device}"):
            for ep in range(cfg.predict.epochs):
                self.model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    b = perm[i:i + bs]
                    opt.zero_grad()
                    pd_, pc_ = self.model(Xtr_t[b])
                    loss = lossf(pd_, ytr_d_t[b])
                    if pc_ is not None:
                        loss = loss + cfg.predict.congestion_loss_weight * lossf(
                            pc_, ytr_c_t[b])
                    loss.backward()
                    opt.step()
                self.model.eval()
                with torch.no_grad():
                    vd, _ = self.model(Xva_t)
                    v = float(lossf(vd, yva_d_t))
                epochs_run = ep + 1
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone()
                                  for k, t in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.predict.patience:
                        LOG.info("  early stop at epoch %d (val MSE %.5f)", ep + 1, best)
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ---- evaluate on the test week ----
        self.model.eval()
        with torch.no_grad():
            pdte, pcte = self.model(torch.tensor(norm(Xte), device=dev))
            pdte = pdte.cpu().numpy()
            pcte = pcte.cpu().numpy() if pcte is not None else None

        pred_counts = np.maximum(np.expm1(np.clip(pdte, 0.0, 12.0)), 0.0)
        true_counts = np.expm1(yte_d)
        mse_counts = float(np.mean((pred_counts - true_counts) ** 2))
        # seasonal-naive baseline: last week's same hour
        base_counts = np.expm1(Xte[:, 0])
        base_mse = float(np.mean((base_counts - true_counts) ** 2))

        if pcte is not None:
            oo, dd, hh, prof = idx_te
            c_true = tt_test.c[oo, dd, prof]
            solid = tt_test.level[oo, dd, prof] == 0
            con_mse = float(np.mean((pcte[solid] - c_true[solid]) ** 2))
            con_mae = float(np.mean(np.abs(pcte[solid] - c_true[solid])))
        else:
            con_mse = con_mae = float("nan")

        res = ForecastResult(
            demand_mse=float(np.mean((pdte - yte_d) ** 2)),
            demand_mae=float(np.mean(np.abs(pdte - yte_d))),
            demand_mse_counts=mse_counts,
            congestion_mse=con_mse, congestion_mae=con_mae,
            baseline_mse_counts=base_mse,
            n_train=len(Xtr), n_test=len(Xte), device=self.device,
            epochs_run=epochs_run)
        LOG.info("forecaster test: demand MSE %.4f (log space), %.2f (counts); "
                 "seasonal-naive %.2f -> %.1f%% better",
                 res.demand_mse, res.demand_mse_counts, res.baseline_mse_counts,
                 100 * (1 - res.demand_mse_counts / max(res.baseline_mse_counts, 1e-9)))
        if np.isfinite(con_mse):
            LOG.info("forecaster test: congestion MSE %.5f, MAE %.4f "
                     "(on directly-measured cells)", con_mse, con_mae)
        self._pairs = pairs
        self._pair_mean = pair_mean
        self._vol = vol
        return res

    # ------------------------------------------------------------------
    def predict_grids(self, cfg: Config, demand, graph, tt_train,
                      hours: np.ndarray):
        """Predicted (demand[o,d,h], congestion[o,d,h]) over the given hours.

        Returned as dense per-hour grids so the MDP can read a lookahead window
        and the Gap 1a utility can read a *predicted* congestion multiplier.
        """
        import torch

        pairs, pm, vol = self._pairs, self._pair_mean, self._vol
        X, _, _, (oo, dd, hh, prof) = build_dataset(
            cfg, demand, graph, tt_train, pairs, hours, pm,
            vol[pairs[:, 0]], vol[pairs[:, 1]])
        dev = torch.device(self.device)
        self.model.eval()
        outs_d, outs_c = [], []
        with torch.no_grad():
            for i in range(0, len(X), 200_000):
                xb = torch.tensor(np.clip((X[i:i + 200_000] - self.mu) / self.sd,
                                          -CLIP, CLIP), device=dev)
                a, b = self.model(xb)
                outs_d.append(a.cpu().numpy())
                if b is not None:
                    outs_c.append(b.cpu().numpy())
        pdem = np.maximum(np.expm1(np.clip(np.concatenate(outs_d), 0.0, 12.0)), 0.0)
        K = graph.n_nodes
        hmap = {int(h): i for i, h in enumerate(hours)}
        dem_grid = np.zeros((K, K, len(hours)), np.float32)
        dem_grid[oo, dd, [hmap[int(h)] for h in hh]] = pdem
        con_grid = None
        if outs_c:
            pcon = np.clip(np.concatenate(outs_c), cfg.utility.c_min,
                           cfg.utility.c_max)
            con_grid = np.ones((K, K, len(hours)), np.float32)
            con_grid[oo, dd, [hmap[int(h)] for h in hh]] = pcon
        return dem_grid, con_grid
