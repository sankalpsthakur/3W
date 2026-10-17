"""Slow or independent implementations that serve as test oracles."""

import numpy as np
from held_out_well.constants import N_CLASSES, OFF_K, ON_K
from held_out_well.detection import part


def alarm_state_reference(inst, elig, score, theta):
    """Plain loop over decisions: the alarm state of the protocol."""
    out = np.zeros(len(score), dtype=bool)
    state = False
    on = off = 0
    for i in range(len(score)):
        if i == 0 or inst[i] != inst[i - 1]:
            state = False
            on = off = 0
        if not elig[i] or not np.isfinite(score[i]):
            state = False
            on = off = 0
            out[i] = False
            continue
        above = theta is not None and score[i] >= theta
        on = on + 1 if above else 0
        off = 0 if above else off + 1
        if on >= ON_K:
            state = True
        if off >= OFF_K:
            state = False
        out[i] = state
    return out


def metrics_numpy(S, classes=None):
    """Closed-form classification, AUROC and AP from summed statistics.

    The formulas of the first implementation of the scorer, written
    with numpy only. The toolkit-based scorer must agree with them.
    """
    S = np.asarray(S, dtype=np.float64)
    out = {}
    conf = part(S, "conf").reshape(S.shape[:-1] + (N_CLASSES, N_CLASSES))
    tp = np.einsum("...ii->...i", conf)
    rowsum, colsum = conf.sum(-1), conf.sum(-2)
    den = 2 * tp + (colsum - tp) + (rowsum - tp)
    with np.errstate(all="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, 0.0)
        rec = np.where(rowsum > 0, tp / rowsum, 0.0)
    present = rowsum > 0
    if classes is not None:
        present = present & np.isin(np.arange(N_CLASSES), classes)
    npres = present.sum(-1)
    safe = np.maximum(npres, 1)
    out["macro_f1"] = np.where(npres > 0, (f1 * present).sum(-1) / safe, np.nan)
    events = present.copy()
    events[..., 0] = False
    nev = events.sum(-1)
    out["macro_f1_event_classes"] = np.where(
        nev > 0, (f1 * events).sum(-1) / np.maximum(nev, 1), np.nan
    )
    out["balanced_accuracy"] = np.where(
        npres > 0, (rec * present).sum(-1) / safe, np.nan
    )
    total = conf.sum((-1, -2))
    out["accuracy"] = np.where(
        total > 0, tp.sum(-1) / np.maximum(total, 1), np.nan
    )
    for c in range(N_CLASSES):
        out[f"f1_c{c}"] = np.where(rowsum[..., c] > 0, f1[..., c], np.nan)
    hp, hn = part(S, "hist_pos"), part(S, "hist_neg")
    P, N = hp.sum(-1), hn.sum(-1)
    with np.errstate(all="ignore"):
        below = np.cumsum(hn, -1) - hn
        auc_num = (hp * (below + 0.5 * hn)).sum(-1)
        out["auroc"] = np.where((P > 0) & (N > 0), auc_num / (P * N), np.nan)
        rp = np.cumsum(hp[..., ::-1], -1)
        rn = np.cumsum(hn[..., ::-1], -1)
        prec = np.where(rp + rn > 0, rp / (rp + rn), 1.0)
        dr = hp[..., ::-1] / np.maximum(P[..., None], 1)
        out["average_precision"] = np.where(P > 0, (prec * dr).sum(-1), np.nan)
    return out
