"""Lineage from measurements: pixel-shift correction, then linking between frames.

find_shifts is the notebook's FFT phase correlation. link_tracks dispatches on link.method:

  "multipass" (default): the notebook's core, in three passes.
      1. Backward, each cell picks the most similar cell in frame t-1 within max_travel, on z-scored
         features. Each mother keeps only its top-scoring child.
      2. Mothers that lost children are checked for division evidence. If it holds, the mother's track
         ends and two children become daughters.
      3. Remaining orphans are matched one-to-one to track ends, across short gaps.
  "notebook": pass 1 alone, where any cell shared by two children is a division.
  "tracks": one-to-one assignment with gap closing, where a track's last `history` detections
      vote on identity, and divisions must conserve mass.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .measure import K, S


def link_features(scalars: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Features compared between detections. (N, M). Log scale: intensities are multiplicative."""
    log2 = lambda v: torch.log2(v.clamp(min=1.0))
    parts = [log2(scalars[:, S["nuc_area"]]).unsqueeze(1), scalars[:, S["eccentricity"]].unsqueeze(1)]
    for name in ("nuc_median", "nuc_std", "cyto_median"):
        parts.append(log2(channels[:, :, K[name]]))
    return torch.cat(parts, dim=1)


def find_shifts(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """FFT phase correlation for frame pairs (N,X,Y). Returns (N,2).

    Same convention as the notebook's find_shift: when content moves by +s from `first` to
    `second`, the result is -s. So x_aligned = x + cumulative shift.
    """
    cross = torch.fft.rfft2(first) * torch.fft.rfft2(second).conj()
    correlation = torch.fft.irfft2(cross / cross.abs().clamp(min=1e-8), s=first.shape[-2:])
    N, X, Y = correlation.shape
    best = correlation.view(N, -1).argmax(dim=1)
    dx, dy = best // Y, best % Y
    dx = torch.where(dx < X / 2, dx, dx - X)
    dy = torch.where(dy < Y / 2, dy, dy - Y)
    return torch.stack([dx, dy], dim=1).float()


def link_features(scalars: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """The notebook's data_array features, without position. (N, M)."""
    log2 = lambda v: torch.log2(v.clamp(min=1.0))
    parts = [scalars[:, [S["edge_n"], S["minor_len"], S["major_len"], S["nuc_area"]]]]
    for name in ("nuc_mean", "nuc_median", "cyto_mean", "cyto_median"):
        parts.append(log2(channels[:, :, K[name]]))
    return torch.cat(parts, dim=1)


def _no_tracks(N: int) -> dict:
    empty = np.zeros(0, dtype=np.int64)
    return {"track_id": np.full(N, -1, dtype=np.int64), "mother_track": empty, "prev_row": np.full(N, -1, dtype=np.int64),
            "mother_row": np.full(N, -1, dtype=np.int64), "drift": np.zeros((1, 2)), "links": 0, "divisions": 0, "tracks": 0}


def estimate_max_travel(xy: torch.Tensor, frames: torch.Tensor, cfg: dict) -> float:
    """How far cells really move per frame, from unambiguous pairs in this movie.

    A pair counts when two cells are mutual nearest neighbours and any runner-up is at least twice
    as far. The gate is the 99th percentile of those moves times a safety factor.
    """
    p = cfg["link"]
    moves = []
    T = int(frames.max().item()) + 1
    for t in range(T - 1):
        a, b = (frames == t).nonzero(as_tuple=True)[0], (frames == t + 1).nonzero(as_tuple=True)[0]
        if a.numel() < 2 or b.numel() < 2:
            continue
        d = torch.cdist(xy[b], xy[a])
        near = d.topk(2, dim=1, largest=False)
        back = d.argmin(dim=0)
        rows = torch.arange(b.numel(), device=xy.device)
        good = (back[near.indices[:, 0]] == rows) & (near.values[:, 1] > 2 * near.values[:, 0]) & (near.values[:, 0] < 150)
        moves.append(near.values[good, 0])
    if not moves or sum(m.numel() for m in moves) < 20:
        return float(p["max_travel_floor"]) * 2
    q99 = torch.quantile(torch.cat(moves), 0.99).item()
    return max(float(p["max_travel_floor"]), q99 * float(p["max_travel_factor"]))


def link_tracks(scalars: torch.Tensor, channels: torch.Tensor, radius: float, cfg: dict, frame_shift: torch.Tensor | None = None) -> dict:
    """Link a timestack's measurements. frame_shift (T,2) comes from find_shifts."""
    N = scalars.shape[0]
    if N == 0 or torch.isnan(scalars[:, S["x"]]).all():
        return _no_tracks(N)
    frames = scalars[:, S["t"]].long()
    T = int(frames.max().item()) + 1
    device = scalars.device
    total = torch.zeros(T, 2, device=device)
    if frame_shift is not None and cfg["link"]["correct_shift"]:
        total = frame_shift[:T].to(device).cumsum(0)  # referenced to frame 0, across batches
    xy = scalars[:, [S["x"], S["y"]]] + total[frames]
    p = cfg["link"]
    max_travel = float(p["max_travel"]) if p["max_travel"] else estimate_max_travel(xy, frames, cfg)
    if p["method"] == "tracks":
        result = _link_with_voting(scalars, channels, xy, radius, cfg, max_travel)
    elif p["method"] == "notebook":
        result = _link_notebook(scalars, channels, xy, cfg, max_travel)
    else:
        result = _link_multipass(scalars, channels, xy, cfg, max_travel)
    result["drift"] = total.cpu().numpy()
    result["max_travel"] = max_travel
    return result


def _zscored_features(scalars, channels, xy) -> torch.Tensor:
    features = torch.cat([xy, link_features(scalars, channels)], dim=1)
    mean = torch.nanmean(features, dim=0)
    std = torch.sqrt(torch.nanmean((features - mean) ** 2, dim=0)).clamp(min=1e-6)
    return (features - mean) / std


def _costs(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """The notebook's similarity: L1 over z-scored features. NaN counts as no difference."""
    return torch.nan_to_num((z_a.unsqueeze(1) - z_b.unsqueeze(0)).abs(), nan=0.0).sum(dim=2)


def _link_multipass(scalars: torch.Tensor, channels: torch.Tensor, xy: torch.Tensor, cfg: dict, max_travel: float) -> dict:
    p = cfg["link"]
    device = scalars.device
    N = scalars.shape[0]
    frames = scalars[:, S["t"]].long()
    z = _zscored_features(scalars, channels, xy)
    rows_of = [(frames == t).nonzero(as_tuple=True)[0] for t in range(int(frames.max().item()) + 1)]
    T = len(rows_of)
    prev_row = torch.full((N,), -1, dtype=torch.long, device=device)
    mother_row = torch.full((N,), -1, dtype=torch.long, device=device)
    prev_cost = torch.full((N,), torch.inf, device=device)

    # ---- Pass 1: each cell picks a mother; each mother keeps its top child.
    loser_rows, loser_mothers = [], []
    for t in range(T - 1, 0, -1):
        now, before = rows_of[t], rows_of[t - 1]
        if now.numel() == 0 or before.numel() == 0:
            continue
        distance = torch.cdist(xy[now], xy[before])
        cost = _costs(z[now], z[before]) + float(p["move_weight"]) * distance / max_travel
        cost[distance > max_travel] = torch.inf
        best_cost, best = cost.min(dim=1)
        found = best_cost < torch.inf
        rows, parents, costs = now[found], before[best[found]], best_cost[found]
        if rows.numel() == 0:
            continue
        order = torch.argsort(costs)  # cheapest first, so the first child seen per mother is the top one
        rows, parents, costs = rows[order], parents[order], costs[order]
        _, inverse = parents.unique(return_inverse=True)
        first = torch.full((inverse.max().item() + 1,), rows.numel(), device=device, dtype=torch.long)
        first = first.scatter_reduce(0, inverse, torch.arange(rows.numel(), device=device), "amin")
        is_top = torch.zeros(rows.numel(), dtype=torch.bool, device=device)
        is_top[first] = True
        prev_row[rows[is_top]] = parents[is_top]
        prev_cost[rows[is_top]] = costs[is_top]
        lost = ~is_top
        loser_rows.append(rows[lost])
        loser_mothers.append(parents[lost])

    pass1_links = int((prev_row >= 0).sum())

    # ---- Pass 2: division evidence for mothers that lost children.
    divisions = 0
    if p["division"] and loser_rows:
        losers = torch.cat(loser_rows)
        if losers.numel():
            divisions = _pass_divisions(scalars, channels, xy, frames, prev_row, mother_row, losers, torch.cat(loser_mothers), cfg)

    # ---- Pass 3: orphans to track ends, one-to-one, across short gaps.
    linked = prev_cost[prev_cost < torch.inf]
    limit = torch.quantile(linked, float(p["cleanup_cost_quantile"])).item() if linked.numel() else float("inf")
    rescued = _pass_cleanup(z, xy, frames, rows_of, prev_row, mother_row, max_travel, limit, cfg)

    track_id, mother_track, _ = _assemble_tracks(prev_row, mother_row, torch.ones(N, dtype=torch.bool, device=device), frames)
    return {"track_id": track_id, "mother_track": mother_track, "prev_row": prev_row.cpu().numpy(), "mother_row": mother_row.cpu().numpy(),
            "links": int((prev_row >= 0).sum()), "divisions": divisions, "tracks": mother_track.size,
            "pass1_links": pass1_links, "cleanup_links": rescued, "cleanup_limit": limit}


def _pass_divisions(scalars, channels, xy, frames, prev_row, mother_row, losers, loser_mothers, cfg) -> int:
    """Accept a division only with evidence: mass conserved, sisters alike, local count up.

    Vectorised: every (mother, top child, losing child) candidate is checked at once, and each
    mother keeps its best-scoring sister. The mother's nuclear mass should split between two
    daughters, so a neighbour that merely picked the same mother fails. Hook for an event classifier.
    """
    p = cfg["link"]
    device = scalars.device
    N = scalars.shape[0]
    mass = channels[:, int(cfg["nuclear_channel"]), K["nuc_sum"]].clamp(min=1.0)
    area = scalars[:, S["nuc_area"]]
    usable = (scalars[:, S["touches_border"]] == 0) & (area >= 0.25 * area.median())

    # The top child of each mother: the row whose prev_row is that mother.
    top_of = torch.full((N,), -1, dtype=torch.long, device=device)
    linked = (prev_row >= 0).nonzero(as_tuple=True)[0]
    top_of[prev_row[linked]] = linked
    top = top_of[loser_mothers]
    ok = (top >= 0) & usable[loser_mothers] & usable[losers] & usable[top.clamp(min=0)]
    total = torch.log2((mass[top.clamp(min=0)] + mass[losers]) / mass[loser_mothers]).abs()
    balance = torch.log2(mass[top.clamp(min=0)] / mass[losers]).abs()
    ok &= (total <= p["division_mass_tol"]) & (balance <= p["division_sister_tol"])

    if p["division_net_count"] and ok.any():
        # Cells within the radius of the mother, in its frame and in the next: the count must rise.
        radius = float(p["division_radius"])
        idx = ok.nonzero(as_tuple=True)[0]
        mothers = loser_mothers[idx]
        rise = torch.zeros(idx.numel(), dtype=torch.bool, device=device)
        mframe = frames[mothers]
        for f in mframe.unique().tolist():
            pick = mframe == f
            centres = xy[mothers[pick]]
            before = (torch.cdist(centres, xy[frames == f]) < radius).sum(1)
            after = (torch.cdist(centres, xy[frames == f + 1]) < radius).sum(1)
            rise[pick] = after > before
        ok[idx] = rise
    if not ok.any():
        return 0

    # One sister per mother: the lowest total + balance.
    score = torch.where(ok, total + balance, torch.full_like(total, torch.inf))
    best = torch.full((N,), torch.inf, device=device).scatter_reduce(0, loser_mothers, score, "amin")
    winner = ok & (score == best[loser_mothers])
    # Ties: keep the first candidate per mother.
    first = torch.full((N,), losers.numel(), dtype=torch.long, device=device)
    first = first.scatter_reduce(0, loser_mothers[winner], winner.nonzero(as_tuple=True)[0], "amin")
    chosen = first[first < losers.numel()]
    mothers, sisters = loser_mothers[chosen], losers[chosen]
    tops = top_of[mothers]
    prev_row[tops] = -1
    mother_row[tops] = mothers
    mother_row[sisters] = mothers
    return int(chosen.numel())


def _pass_cleanup(z, xy, frames, rows_of, prev_row, mother_row, max_travel, limit, cfg) -> int:
    """Forward in time: each orphan may claim an unused track end, within the gate and the gap."""
    p = cfg["link"]
    device = z.device
    N = z.shape[0]
    has_child = torch.zeros(N, dtype=torch.bool, device=device)
    has_child[prev_row[prev_row >= 0]] = True
    has_child[mother_row[mother_row >= 0]] = True
    max_gap = int(p["max_gap"])
    penalty = float(p["gap_penalty"])
    rescued = 0
    T = len(rows_of)
    for t in range(1, T):
        now = rows_of[t]
        if now.numel() == 0:
            continue
        orphans = now[(prev_row[now] < 0) & (mother_row[now] < 0)]
        if orphans.numel() == 0:
            continue
        earlier = [rows_of[g] for g in range(max(0, t - 1 - max_gap), t)]
        ends = torch.cat(earlier) if earlier else orphans[:0]
        ends = ends[~has_child[ends]]
        if ends.numel() == 0:
            continue
        gap = (t - frames[ends]).float()
        distance = torch.cdist(xy[orphans], xy[ends])
        cost = _costs(z[orphans], z[ends]) + penalty * (gap - 1) + float(p["move_weight"]) * distance / max_travel
        allowed = (distance <= max_travel * (1 + 0.5 * (gap - 1))) & (cost <= limit)
        if not allowed.any():
            continue
        matrix = torch.where(allowed, cost, torch.full_like(cost, 1e6)).cpu().numpy()
        r, c = linear_sum_assignment(matrix)
        good = matrix[r, c] < 1e6
        r, c = r[good], c[good]
        if r.size:
            prev_row[orphans[torch.as_tensor(r, device=device)]] = ends[torch.as_tensor(c, device=device)]
            has_child[ends[torch.as_tensor(c, device=device)]] = True
            rescued += int(r.size)
    return rescued


def _link_notebook(scalars: torch.Tensor, channels: torch.Tensor, xy: torch.Tensor, cfg: dict, max_travel: float) -> dict:
    """The notebook's lineage cell. Backward through time, on globally z-scored features.

    Differences from the cell, each needed for it to run correctly: argmin is taken over t0 for
    each t1 cell (the cell mixed the axes), a cell with no candidate starts a new track (the cell
    linked it to row 0), NaN features count as zero difference, and if 2+ cells pick one mother
    they are its daughters only when division is enabled; otherwise the closest one keeps the track.
    """
    p = cfg["link"]
    device = scalars.device
    N = scalars.shape[0]
    frames = scalars[:, S["t"]].long()
    z = _zscored_features(scalars, channels, xy)
    prev_row = torch.full((N,), -1, dtype=torch.long, device=device)
    mother_row = torch.full((N,), -1, dtype=torch.long, device=device)
    rows_of = [(frames == t).nonzero(as_tuple=True)[0] for t in range(int(frames.max().item()) + 1)]
    links = divisions = 0
    for t in range(len(rows_of) - 1, 0, -1):
        now, before = rows_of[t], rows_of[t - 1]
        if now.numel() == 0 or before.numel() == 0:
            continue
        distance = torch.cdist(xy[now], xy[before])
        cost = torch.nan_to_num((z[now].unsqueeze(1) - z[before].unsqueeze(0)).abs(), nan=0.0).sum(dim=2)
        cost[distance > max_travel] = torch.inf
        best = cost.argmin(dim=1)
        found = cost.gather(1, best[:, None])[:, 0] < torch.inf
        rows, parents = now[found], before[best[found]]
        unique, counts = parents.unique(return_counts=True)
        shared = unique[counts >= 2]
        single = ~torch.isin(parents, shared)
        prev_row[rows[single]] = parents[single]
        links += int(single.sum())
        for mother in shared.tolist():
            kids = rows[parents == mother]
            if p["division"]:
                mother_row[kids] = mother
                divisions += 1
            else:
                closest = kids[torch.cdist(xy[kids], xy[mother][None])[:, 0].argmin()]
                prev_row[closest] = mother
                links += 1
    track_id, mother_track, _ = _assemble_tracks(prev_row, mother_row, torch.ones(N, dtype=torch.bool, device=device), frames)
    return {"track_id": track_id, "mother_track": mother_track, "prev_row": prev_row.cpu().numpy(), "mother_row": mother_row.cpu().numpy(),
            "links": links, "divisions": divisions, "tracks": mother_track.size}


def estimate_scales(xy, features, frames, gate: float) -> tuple[torch.Tensor, float]:
    """Per-frame change of the same cell: per-feature scale and a motion scale.

    Pairs are confident if mutually nearest and the runner-up is at least twice as far.
    """
    T = int(frames.max().item()) + 1
    diffs, moves = [], []
    for t in range(T - 1):
        a, b = (frames == t).nonzero(as_tuple=True)[0], (frames == t + 1).nonzero(as_tuple=True)[0]
        if a.numel() < 2 or b.numel() < 2:
            continue
        d = torch.cdist(xy[b], xy[a])
        near = d.topk(2, dim=1, largest=False)
        back = d.argmin(dim=0)
        rows = torch.arange(b.numel(), device=xy.device)
        good = (back[near.indices[:, 0]] == rows) & (near.values[:, 0] < gate) & (near.values[:, 1] > 2 * near.values[:, 0])
        if good.any():
            diffs.append(features[b[good]] - features[a[near.indices[good, 0]]])
            moves.append(near.values[good, 0])
    if not diffs:
        return torch.ones(features.shape[1], device=xy.device), max(gate / 4, 1.0)
    delta = torch.cat(diffs)
    scales = (1.4826 * (delta - delta.nanmedian(dim=0).values).abs().nanmedian(dim=0).values).clamp(min=1e-3)
    motion = 1.4826 * torch.cat(moves).median().item()
    return scales, max(motion, 0.5)


def _assign_gpu(i: torch.Tensor, j: torch.Tensor, cost: torch.Tensor, n: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterative mutual best on sparse pairs. Close to optimal when cost scales are sane."""
    keep_i, keep_j = [], []
    alive = torch.ones(i.shape, dtype=torch.bool, device=i.device)
    inf = torch.tensor(float("inf"), device=i.device)
    while alive.any():
        c = torch.where(alive, cost, inf)
        best_i = torch.full((n,), float("inf"), device=i.device).scatter_reduce(0, i, c, "amin")
        best_j = torch.full((m,), float("inf"), device=i.device).scatter_reduce(0, j, c, "amin")
        mutual = alive & (c == best_i[i]) & (c == best_j[j])
        if not mutual.any():
            break
        keep_i.append(i[mutual])
        keep_j.append(j[mutual])
        used_i, used_j = torch.zeros(n, dtype=torch.bool, device=i.device), torch.zeros(m, dtype=torch.bool, device=i.device)
        used_i[i[mutual]] = True
        used_j[j[mutual]] = True
        alive = alive & ~used_i[i] & ~used_j[j]
    if not keep_i:
        empty = torch.zeros(0, dtype=torch.long, device=i.device)
        return empty, empty
    return torch.cat(keep_i), torch.cat(keep_j)


def _assign_lsa(i, j, cost, n, m, max_cost: float):
    """Exact assignment on the CPU, with `max_cost` as the price of leaving a pair unmatched."""
    forbid = 1e6
    matrix = np.full((n, m), forbid, dtype=np.float64)
    matrix[i.cpu().numpy(), j.cpu().numpy()] = cost.cpu().numpy()
    padded = np.full((n + m, n + m), 0.0)
    padded[:n, :m] = matrix
    padded[:n, m:] = forbid
    padded[n:, :m] = forbid
    padded[np.arange(n), m + np.arange(n)] = max_cost
    padded[n + np.arange(m), np.arange(m)] = max_cost
    r, c = linear_sum_assignment(padded)
    good = (r < n) & (c < m)
    r, c = r[good], c[good]
    ok = matrix[r, c] < forbid
    return torch.as_tensor(r[ok], device=i.device), torch.as_tensor(c[ok], device=i.device)


def _history(prev_row: torch.Tensor, ends: torch.Tensor, K_hist: int) -> torch.Tensor:
    """Rows of each track's last detections. (E, K); -1 where the track is shorter."""
    hist = torch.full((ends.numel(), K_hist), -1, dtype=torch.long, device=ends.device)
    current = ends.clone()
    for k in range(K_hist):
        hist[:, k] = current
        valid = current >= 0
        nxt = torch.full_like(current, -1)
        nxt[valid] = prev_row[current[valid]]
        current = nxt
    return hist


def _link_with_voting(scalars: torch.Tensor, channels: torch.Tensor, xy: torch.Tensor, radius: float, cfg: dict, max_travel: float) -> dict:
    """One-to-one matching to active tracks, with a multi-frame identity vote."""
    p = cfg["link"]
    device = scalars.device
    N = scalars.shape[0]
    valid = ~torch.isnan(xy[:, 0])
    frames = scalars[:, S["t"]].long()
    gate = max_travel
    T = int(frames.max().item()) + 1
    features = link_features(scalars, channels)
    scales, motion = estimate_scales(xy[valid], features[valid], frames[valid], gate)
    z = features / scales

    prev_row = torch.full((N,), -1, dtype=torch.long, device=device)
    mother_row = torch.full((N,), -1, dtype=torch.long, device=device)
    K_hist = int(p["history"])
    clip = float(p["feature_clip"])
    max_gap = int(p["max_gap"])
    active = torch.zeros(0, dtype=torch.long, device=device)
    n_links = n_divisions = 0
    nuclear = int(cfg["nuclear_channel"])
    mass = channels[:, nuclear, K["nuc_sum"]].clamp(min=1.0)
    # An edge sliver or a tiny fragment is not a daughter cell.
    area = scalars[:, S["nuc_area"]]
    can_divide = (scalars[:, S["touches_border"]] == 0) & (area >= 0.25 * area[valid].median())

    for t in range(T):
        current = ((frames == t) & valid).nonzero(as_tuple=True)[0]
        if current.numel() == 0:
            continue
        active = active[frames[active] >= t - 1 - max_gap]
        unmatched_cells = torch.ones(current.numel(), dtype=torch.bool, device=device)
        matched_end = torch.full((current.numel(),), -1, dtype=torch.long, device=device)
        if active.numel():
            prior = prev_row[active]
            gap = (t - frames[active]).float()
            velocity = torch.zeros_like(xy[active])
            has = prior >= 0
            velocity[has] = (xy[active[has]] - xy[prior[has]]) / (frames[active[has]] - frames[prior[has]]).unsqueeze(1).float()
            predicted = xy[active] + 0.5 * velocity * gap.unsqueeze(1)
            dist = torch.cdist(xy[current], predicted)
            allowed = dist <= gate * (1 + 0.5 * (gap - 1)).unsqueeze(0)
            i, j = allowed.nonzero(as_tuple=True)
            if i.numel():
                hist = _history(prev_row, active, K_hist)[j]
                delta = (z[current[i]].unsqueeze(1) - z[hist.clamp(min=0)]).abs().clamp(max=clip)
                dissim = torch.nan_to_num(delta, nan=0.0).mean(-1)  # (P, K)
                dissim = torch.where(hist >= 0, dissim, torch.nan)
                vote = dissim.nanmedian(dim=1).values
                cost = vote + p["distance_weight"] * dist[i, j] / motion + p["gap_penalty"] * (gap[j] - 1)
                allowed_cost = cost <= p["max_cost"]
                i, j, cost = i[allowed_cost], j[allowed_cost], cost[allowed_cost]
                if i.numel():
                    # Exact assignment is cubic; above this size the GPU matcher is nearly as good.
                    if p["assignment"] == "lsa" and max(current.numel(), active.numel()) <= int(p["lsa_max_cells"]):
                        mi, mj = _assign_lsa(i, j, cost, current.numel(), active.numel(), float(p["max_cost"]))
                    else:
                        mi, mj = _assign_gpu(i, j, cost, current.numel(), active.numel())
                    prev_row[current[mi]] = active[mj]
                    unmatched_cells[mi] = False
                    matched_end[mi] = active[mj]
                    n_links += mi.numel()

        if p["division"] and unmatched_cells.any() and active.numel():
            n_divisions += _find_divisions(current, unmatched_cells, matched_end, active, frames, xy, mass, can_divide, prev_row,
                                           mother_row, t, float(p["division_distance"]),
                                           float(p["division_mass_tol"]), float(p["division_sister_tol"]))

        succeeded = torch.zeros(N, dtype=torch.bool, device=device)
        linked = prev_row[current]
        succeeded[linked[linked >= 0]] = True
        mothers = mother_row[current]
        succeeded[mothers[mothers >= 0]] = True
        active = torch.cat([active[~succeeded[active]], current])

    track_id, mother_track, order = _assemble_tracks(prev_row, mother_row, valid, frames)
    return {
        "track_id": track_id, "mother_track": mother_track, "prev_row": prev_row.cpu().numpy(),
        "mother_row": mother_row.cpu().numpy(), "scales": scales.cpu().numpy(),
        "motion_scale": motion, "links": n_links, "divisions": n_divisions, "tracks": mother_track.size,
    }


def _find_divisions(current, unmatched, matched_end, active, frames, xy, mass, can_divide, prev_row, mother_row, t, reach, mass_tol, sister_tol) -> int:
    """Pair each new cell with a sister and a mother from the frame before. Mass is conserved
    across division: the two daughters sum to about the mother, and each is about half."""
    device = current.device
    mothers = active[frames[active] == t - 1]
    mothers = mothers[can_divide[mothers]]
    if mothers.numel() == 0:
        return 0
    found = 0
    new_rows = current[unmatched]
    new_rows = new_rows[can_divide[new_rows]]
    if new_rows.numel() == 0:
        return 0
    d = torch.cdist(xy[new_rows], xy[mothers])
    # Candidate sisters: other new cells, and cells that have just linked to this mother.
    taken = set()
    for k in range(new_rows.numel()):
        row = new_rows[k]
        if int(row) in taken:
            continue
        candidates = (d[k] <= reach).nonzero(as_tuple=True)[0]
        best = None
        for c in candidates.tolist():
            mother = mothers[c]
            sisters = [r for r in current[matched_end == mother].tolist() if can_divide[r]]
            sisters += [int(r) for r in new_rows.tolist() if int(r) != int(row) and int(r) not in taken
                        and torch.dist(xy[r], xy[mother]) <= reach]
            for sister in sisters:
                total = torch.log2((mass[row] + mass[sister]) / mass[mother]).abs().item()
                balance = torch.log2(mass[row] / mass[sister]).abs().item()
                if total <= mass_tol and balance <= sister_tol:
                    score = total + balance + float(d[k, c]) / reach
                    if best is None or score < best[0]:
                        best = (score, int(mother), int(sister))
        if best is None:
            continue
        _, mother, sister = best
        for daughter in (int(row), sister):
            prev_row[daughter] = -1
            mother_row[daughter] = mother
            taken.add(daughter)
        found += 1
    return found


def _assemble_tracks(prev_row, mother_row, valid, frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give each row a track id by walking back along prev_row. Vectorised path compression."""
    prev = prev_row.cpu().numpy()
    root = np.arange(prev.size)
    linked = prev >= 0
    root[linked] = prev[linked]
    while True:
        jumped = root[root]
        if np.array_equal(jumped, root):
            break
        root = jumped
    is_valid = valid.cpu().numpy()
    roots, track_id = np.unique(root[is_valid], return_inverse=True)
    full = np.full(prev.size, -1, dtype=np.int64)
    full[is_valid] = track_id
    mother = mother_row.cpu().numpy()
    mother_track = np.full(roots.size, -1, dtype=np.int64)
    daughters = (mother >= 0) & is_valid
    mother_track[full[daughters]] = full[mother[daughters]]
    return full, mother_track, roots


def trace_index(track_id: np.ndarray, frames: np.ndarray, T: int) -> np.ndarray:
    """(tracks, T) int32 of measurement rows, -1 where absent. Any metric becomes a matrix by indexing."""
    n_tracks = int(track_id.max()) + 1 if track_id.size else 0  # -1 alone also gives zero tracks
    index = np.full((n_tracks, T), -1, dtype=np.int32)
    ok = track_id >= 0
    index[track_id[ok], frames[ok].astype(int)] = np.nonzero(ok)[0]
    return index
