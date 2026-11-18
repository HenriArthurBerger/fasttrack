"""Score pipeline output against synthetic ground truth."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def match_detections(det_t, det_x, det_y, truth: dict, size: int, gate: float, signed_labels=None, det_labels=None) -> dict:
    """Optimal per-frame match of detections to true nuclei.

    Returns det_cell (detection row -> true cell id, or -1), scores, and undersegmentation data.
    """
    det_cell = np.full(det_t.shape, -1, np.int64)
    border = 3
    in_frame = (truth["x"] >= border) & (truth["x"] < size - border) & (truth["y"] >= border) & (truth["y"] < size - border)
    tt, tx, ty, tc = truth["t"][in_frame], truth["x"][in_frame], truth["y"][in_frame], truth["cell"][in_frame]
    tp = fp = fn = 0
    merged_rows = np.zeros(det_t.shape, bool)
    gt_in_merge = 0
    for t in np.unique(tt):
        d_rows = np.nonzero(det_t == t)[0]
        g_rows = np.nonzero(tt == t)[0]
        if d_rows.size == 0:
            fn += g_rows.size
            continue
        dpts = np.stack([det_x[d_rows], det_y[d_rows]], 1)
        gpts = np.stack([tx[g_rows], ty[g_rows]], 1)
        cost = np.linalg.norm(dpts[:, None] - gpts[None], axis=2)
        capped = np.where(cost > gate, 1e6, cost)
        ri, ci = linear_sum_assignment(capped)
        good = cost[ri, ci] <= gate
        det_cell[d_rows[ri[good]]] = tc[g_rows[ci[good]]]
        tp += int(good.sum())
        fp += d_rows.size - int(good.sum())
        fn += g_rows.size - int(good.sum())
        if signed_labels is not None:
            frame = signed_labels[int(t), 0]
            hit = frame[np.clip(np.round(gpts[:, 0]).astype(int), 0, size - 1), np.clip(np.round(gpts[:, 1]).astype(int), 0, size - 1)]
            hit = hit[hit > 0]
            ids, counts = np.unique(hit, return_counts=True)
            multi = ids[counts >= 2]
            gt_in_merge += int(counts[counts >= 2].sum())
            if det_labels is not None and multi.size:
                merged_rows[d_rows[np.isin(det_labels[d_rows], multi)]] = True  # labels are local to a frame
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "det_cell": det_cell, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "merged_rows": merged_rows, "merged_detections": int(merged_rows.sum()),
        "gt_in_merge": gt_in_merge, "gt_total": int(in_frame.sum()),
    }


def instance_iou(signed_labels: np.ndarray, truth_labels: np.ndarray, frames=None) -> float:
    """Mean best-match IoU over true nuclei, using nuclear pixels only."""
    frames = range(truth_labels.shape[0]) if frames is None else frames
    ious = []
    for t in frames:
        gt = truth_labels[t]
        pred = np.maximum(signed_labels[t, 0], 0)
        pairs = (gt > 0) & (pred > 0)
        gt_ids, gt_area = np.unique(gt[gt > 0], return_counts=True)
        pr_ids, pr_area = np.unique(pred[pred > 0], return_counts=True)
        area_gt, area_pr = dict(zip(gt_ids, gt_area)), dict(zip(pr_ids, pr_area))
        pair_key, overlap = np.unique(gt[pairs].astype(np.int64) * (2**31) + pred[pairs].astype(np.int64), return_counts=True)
        best: dict[int, float] = {}
        for key, inter in zip(pair_key, overlap):
            g, p = int(key // 2**31), int(key % 2**31)
            best[g] = max(best.get(g, 0.0), inter / (area_gt[g] + area_pr[p] - inter))
        ious.extend(best.get(int(g), 0.0) for g in gt_ids)
    return float(np.mean(ious)) if ious else 0.0


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC. NaN scores are dropped."""
    ok = ~np.isnan(score)
    score, positive = score[ok], positive[ok]
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(score.size)
    ranks[order] = np.arange(1, score.size + 1)
    uniq, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def count_true_links(truth: dict, size: int, border: int = 3) -> int:
    """How many same-cell links between consecutive frames exist among in-frame true nuclei."""
    inside = (truth["x"] >= border) & (truth["x"] < size - border) & (truth["y"] >= border) & (truth["y"] < size - border)
    keys = truth["t"][inside].astype(np.int64) * (2**31) + truth["cell"][inside]
    return int(np.isin(keys + 2**31, keys).sum())


def evaluate_tracking(track_id: np.ndarray, det_t: np.ndarray, det_cell: np.ndarray, truth: dict, size: int | None = None) -> dict:
    """Link-level accuracy of a tracking result against true cell identities.

    A true link is the same cell, or mother to daughter, in consecutive frames. It is correct
    when both ends were detected and carry the same track id (or a declared division).
    """
    matched = det_cell >= 0
    key = det_t[matched].astype(np.int64) * (2**31) + det_cell[matched]
    tid = track_id[matched]
    lookup = dict(zip(key.tolist(), tid.tolist()))
    good = bad = missed = 0
    for t, cell in zip(det_t[matched], det_cell[matched]):
        prev = lookup.get(int(t - 1) * (2**31) + int(cell))
        if prev is None:
            missed += 1
            continue
        here = lookup[int(t) * (2**31) + int(cell)]
        if here == prev:
            good += 1
        else:
            bad += 1
    # Tracks spanning the most of each true cell's life (identity switches, whole-track view)
    per_cell: dict[int, dict[int, int]] = {}
    for cell, track in zip(det_cell[matched].tolist(), tid.tolist()):
        d = per_cell.setdefault(cell, {})
        d[track] = d.get(track, 0) + 1
    purity_num = sum(max(d.values()) for d in per_cell.values())
    purity_den = sum(sum(d.values()) for d in per_cell.values())
    per_track: dict[int, dict[int, int]] = {}
    for cell, track in zip(det_cell[matched].tolist(), tid.tolist()):
        d = per_track.setdefault(track, {})
        d[cell] = d.get(cell, 0) + 1
    t_num = sum(max(d.values()) for d in per_track.values())
    t_den = sum(sum(d.values()) for d in per_track.values())
    fragments = np.mean([len(d) for d in per_cell.values()]) if per_cell else 0.0
    all_links = count_true_links(truth, size) if size else 0
    return {
        # True links recovered, with missed cells counted as failures. This one cannot be inflated by merging.
        "true_link_recall": good / all_links if all_links else float("nan"),
        "link_accuracy": good / max(good + bad, 1), "id_breaks": bad, "links_correct": good, "unlinkable": missed,
        "cell_coverage": purity_num / max(purity_den, 1), "track_purity": t_num / max(t_den, 1),
        "fragments_per_cell": float(fragments), "true_cells": len(per_cell), "tracks": len(per_track),
    }


def evaluate_divisions(track_id, det_t, det_cell, truth, mother_track: np.ndarray) -> dict:
    """Division recall and precision. A declared daughter counts if the true cell at its first
    frame really is a newborn and its declared mother track mostly follows the true parent."""
    parent_of = {c: v["parent"] for c, v in truth["tracks"].items()}
    start_of = {c: v["start"] for c, v in truth["tracks"].items()}
    majority: dict[int, int] = {}
    counts: dict[int, dict[int, int]] = {}
    for tr, cell in zip(track_id.tolist(), det_cell.tolist()):
        if cell >= 0:
            d = counts.setdefault(tr, {})
            d[cell] = d.get(cell, 0) + 1
    for tr, d in counts.items():
        majority[tr] = max(d, key=d.get)
    true_daughters = {c for c, p in parent_of.items() if p >= 0 and c in set(det_cell.tolist())}
    found = set()
    declared = correct = 0
    for tr, mother in enumerate(mother_track.tolist()):
        if mother < 0:
            continue
        declared += 1
        cell = majority.get(tr, -1)
        if cell >= 0 and parent_of.get(cell, -1) >= 0 and majority.get(mother, -2) == parent_of[cell]:
            correct += 1
            found.add(cell)
    return {"true_daughters": len(true_daughters), "declared_daughters": declared, "correct_daughters": correct,
            "division_recall": len(found) / max(len(true_daughters), 1), "division_precision": correct / max(declared, 1)}
