"""Lineage from measurements, over a whole timestack. Touches no pixels, so it costs little.

Each cell in frame t is matched to the end of an active track. A pair's cost is:

    identity vote + distance + gap penalty

Identity vote: the cell is compared to each of the track's last `history` detections, and the
median of those dissimilarities is the vote. One bad frame (a merge, a focus blip) is outvoted
by the rest of the track, which frame-to-frame matching cannot do.

Features are scaled by how much the same cell changes per frame, measured from confident
pairs in this movie. A feature that is stable within a cell and varies between cells counts
for the most. No weights need tuning per dataset.

Result: track ids plus a mother for each daughter. Tracks end at division; both daughters
start new tracks.
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


def estimate_drift(xy: torch.Tensor, frames: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Cumulative stage shift per frame (T, 2), by a vote over all pairwise displacements.

    Real displacement between frames puts many cell pairs in one bin; chance pairs scatter.
    Works from centroids, so it needs no images and crosses batch boundaries freely.
    """
    p = cfg["link"]
    T = int(frames.max().item()) + 1
    shift = torch.zeros(T, 2, device=xy.device)
    search, bin_px = float(p["drift_search_px"]), float(p["drift_bin_px"])
    n_bins = int(2 * search / bin_px) + 1
    previous = xy[frames == 0]
    for t in range(1, T):
        current = xy[frames == t]
        if previous.shape[0] >= 3 and current.shape[0] >= 3:
            delta = (current[None] - previous[:, None]).view(-1, 2)
            delta = delta[(delta.abs() < search).all(1)]
            if delta.numel():
                idx = ((delta + search) / bin_px).long()
                hist = torch.bincount(idx[:, 0] * n_bins + idx[:, 1], minlength=n_bins * n_bins).view(1, 1, n_bins, n_bins).float()
                hist = torch.nn.functional.avg_pool2d(hist, 3, stride=1, padding=1) * 9  # votes on a bin edge
                best = int(hist.argmax().item())
                peak = (torch.tensor([best // n_bins, best % n_bins], device=xy.device).float() + 0.5) * bin_px - search
                near = delta[((delta - peak).abs() <= 1.5 * bin_px).all(1)]
                enough = near.shape[0] >= max(3, 0.4 * min(previous.shape[0], current.shape[0]))
                if enough:
                    estimate = near.median(dim=0).values
                    if estimate.norm() >= p["drift_min_px"]:
                        shift[t] = estimate
        previous = current
    return shift.cumsum(0)


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


def link_tracks(scalars: torch.Tensor, channels: torch.Tensor, radius: float, cfg: dict) -> dict:
    """Returns numpy arrays: track_id (N), mother_track (tracks), plus drift, scales and counts."""
    p = cfg["link"]
    device = scalars.device
    N = scalars.shape[0]
    if N == 0 or torch.isnan(scalars[:, S["x"]]).all():
        empty = np.zeros(0, dtype=np.int64)
        return {"track_id": np.full(N, -1, dtype=np.int64), "mother_track": empty, "prev_row": np.full(N, -1, dtype=np.int64),
                "mother_row": np.full(N, -1, dtype=np.int64), "drift": np.zeros((1, 2)), "scales": np.zeros(0),
                "motion_scale": 0.0, "links": 0, "divisions": 0, "tracks": 0}
    valid = ~torch.isnan(scalars[:, S["x"]])
    frames = scalars[:, S["t"]].long()
    xy_raw = scalars[:, [S["x"], S["y"]]]
    gate = float(p["max_travel_r"] * radius)

    drift = torch.zeros(int(frames.max().item()) + 1, 2, device=device)
    if p["correct_drift"]:
        drift = estimate_drift(xy_raw[valid], frames[valid], cfg)
    xy = xy_raw - drift[frames]
    features = link_features(scalars, channels)
    scales, motion = estimate_scales(xy[valid], features[valid], frames[valid], gate)
    z = features / scales

    prev_row = torch.full((N,), -1, dtype=torch.long, device=device)
    mother_row = torch.full((N,), -1, dtype=torch.long, device=device)
    K_hist = int(p["history"])
    clip = float(p["feature_clip"])
    max_gap = int(p["max_gap"])
    T = drift.shape[0]
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
                                           mother_row, t, float(p["division_distance_r"] * radius),
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
        "mother_row": mother_row.cpu().numpy(), "drift": drift.cpu().numpy(), "scales": scales.cpu().numpy(),
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
