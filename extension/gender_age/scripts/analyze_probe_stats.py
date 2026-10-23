"""Small statistics helpers shared by the analyses (no heavy imports)."""
import numpy as np


def p_boot(d):
    """Two-sided bootstrap p-value for 'difference = 0'."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return np.nan
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
