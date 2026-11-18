"""Per-cell measurements: `measure_subsections` from the notebook, plus packaging.

The frame is cut into overlapping sub-sections. Local labels say which section owns each
pixel, so all cells of a section are measured at once with scatter operations.

measure_subsections returns (T, C, subsections**2, section_max, NUM_METRICS).
measure_batch flattens that to the saved form: scalars (N, S) and channels (N, C, K).
"""
from __future__ import annotations

import math
from enum import IntEnum

import torch

class M(IntEnum):
    """Metric axis layout for the output tensor's last dimension.

    Slots 0-19 are unchanged from the original module (backward compatible).
    Slots 20+ are geometric / merge-detection features.

    Per-channel metrics vary along the C axis of the output. Everything else is
    a cell-level scalar broadcast across channels.
    """
    # --- nuclear intensity (per channel) ---
    NUC_MEAN            = 0
    NUC_MEDIAN          = 1
    NUC_SUM             = 2
    NUC_STD             = 3
    NUC_MIN             = 4
    NUC_MAX             = 5
    NUC_N               = 6    # broadcast across channels (presence key)
    # --- cytoplasmic intensity (per channel) ---
    CYTO_MEAN           = 7
    CYTO_MEDIAN         = 8
    CYTO_SUM            = 9
    CYTO_STD            = 10
    CYTO_MIN            = 11
    CYTO_MAX            = 12
    CYTO_N              = 13   # broadcast across channels
    # --- ratios / geometry (original) ---
    EDGE_N              = 14   # broadcast: perimeter (border pixel count)
    NC_MEAN_RATIO       = 15   # per channel: nuc_mean / cyto_mean
    NUC_SUM_FRAC        = 16   # per channel: nuc_sum / (nuc_sum + cyto_sum)
    CENTROID_X          = 17   # broadcast
    CENTROID_Y          = 18   # broadcast
    CELL_LABEL          = 19   # broadcast
    # --- nuclear shape (all broadcast across channels) ---
    NUC_AREA            = 20   # nuclear pixel count (explicit; == NUC_N)
    NUC_ECCENTRICITY    = 21
    NUC_ELONGATION      = 22
    NUC_ORIENTATION     = 23
    NUC_MAJOR_LEN       = 24
    NUC_MINOR_LEN       = 25
    NUC_ELLIPSE_RATIO   = 26   # area / equivalent-ellipse area
    NUC_EXTENT          = 27   # area / bbox area
    NUC_BBOX_ASPECT     = 28   # bbox height / width
    NUC_RADIAL_ROUGH    = 29   # std(edge->centroid r) / mean(r)
    NUC_CIRCULARITY     = 30   # 4*pi*area / perimeter^2
    NUC_HU1             = 31   # 1st Hu moment invariant (rot/scale invariant)
    NUC_HU2             = 32   # 2nd Hu moment invariant
    CONCENTRICITY       = 33   # |nuc centroid - cyto centroid| / r_nuc
    # --- multichannel merge signals ---
    CHAN_CENTROID_OFF   = 34   # per channel: intensity-weighted centroid divergence
    NUC_BIMODALITY      = 35   # per channel: Sarle's bimodality coefficient
    CORR_01             = 36   # within-nucleus Pearson r, channels (0,1)
    CORR_02             = 37   # within-nucleus Pearson r, channels (0,2)
    CORR_12             = 38   # within-nucleus Pearson r, channels (1,2)
    # --- crowding (filled by a post-step, NaN inside the pixel loop) ---
    LOCAL_DENSITY       = 39   # # other nuclear centroids within 2*r_nuc


NUM_METRICS = len(M)

# Correlation channel pairs. Names and pairs now agree: (0,1)->CORR_01,
# (0,2)->CORR_02, (1,2)->CORR_12. Guarded against C < 3 at runtime.
_CORR_PAIRS = [(0, 1), (0, 2), (1, 2)]
_CORR_SLOTS = [M.CORR_01, M.CORR_02, M.CORR_12]


def measure_subsections(data_tensor, label_map, border_labels,
                        section_max, radius, subsections=7,
                        even_median='lower', density_post=True):
    """
    data_tensor:   (T, C, X, Y) int16 on cuda
    label_map:     (T, 1, X, Y) signed: +label = nucleus, -label = cytoplasm.
                   Label encodes subsection as (abs(label) // 1000) == sidx and
                   cell id as (abs(label) % 1000) - 1. Requires section_max < 1000.
    border_labels: (T, 1, X, Y) +label on the edge ring
    density_post:  if True, fill LOCAL_DENSITY from centroids after the loop.
    Returns (T, C, subsections**2, section_max, NUM_METRICS) float32.
    """
    T, C, X, Y = data_tensor.shape
    dev = data_tensor.device

    out = torch.full((T, C, subsections**2, section_max, NUM_METRICS),
                     float('nan'), dtype=torch.float32, device=dev)

    xbin_size = X // subsections + 1
    ybin_size = Y // subsections + 1

    for sub_x in range(subsections):
        for sub_y in range(subsections):
            sidx = sub_y * subsections + sub_x

            lx = max(0, sub_x * xbin_size - 3 * radius)
            hx = min(X, (sub_x + 1) * xbin_size + 3 * radius)
            ly = max(0, sub_y * ybin_size - 3 * radius)
            hy = min(Y, (sub_y + 1) * ybin_size + 3 * radius)

            lab    = label_map[:, :, lx:hx, ly:hy].squeeze(1)      # (T,h,w)
            border = border_labels[:, :, lx:hx, ly:hy].squeeze(1)  # (T,h,w)
            data   = data_tensor[:, :, lx:hx, ly:hy]               # (T,C,h,w) int16

            # ---- gather pixels belonging to this subsection's cells ----
            owned = (lab != 0) & ((lab.abs() // 1000) == sidx)
            if not owned.any():
                continue

            nz = owned.nonzero(as_tuple=False)              # (N,3): t,i,j
            t_i, i_i, j_i = nz[:, 0], nz[:, 1], nz[:, 2]
            lv = lab[t_i, i_i, j_i]                         # signed label
            vals = data[t_i, :, i_i, j_i].float()           # (N,C)

            cid = (lv.abs() % 1000).long() - 1              # 0..section_max-1
            valid = (cid >= 0) & (cid < section_max)
            if not valid.all():
                t_i, i_i, j_i = t_i[valid], i_i[valid], j_i[valid]
                lv, vals, cid = lv[valid], vals[valid], cid[valid]

            is_cyto = (lv < 0).long()

            # ===================================================================
            # G-space segments (t, cell, nuc|cyto) : original intensity stats
            # ===================================================================
            seg = (t_i * section_max + cid) * 2 + is_cyto
            G = T * section_max * 2
            ones = torch.ones(seg.shape[0], device=dev)

            cnt = torch.zeros(G, device=dev).index_add_(0, seg, ones)
            # FIX: accumulate sum / sumsq in float64 so large nuclear sums stay
            # exact (float32 loses precision past ~1.6e7).
            s1  = torch.zeros(G, C, device=dev, dtype=torch.float64).index_add_(
                0, seg, vals.double())
            s2  = torch.zeros(G, C, device=dev, dtype=torch.float64).index_add_(
                0, seg, (vals * vals).double())

            vmin_t = torch.zeros(G, C, device=dev).index_reduce_(
                0, seg, vals, 'amin', include_self=False)
            vmax_t = torch.zeros(G, C, device=dev).index_reduce_(
                0, seg, vals, 'amax', include_self=False)

            # edge pixel counts (nuc slot) — also serves as perimeter
            bmask = (border != 0) & ((border.abs() // 1000) == sidx)
            if bmask.any():
                bz = bmask.nonzero(as_tuple=False)
                bcid = (border[bz[:, 0], bz[:, 1], bz[:, 2]].abs() % 1000).long() - 1
                bok = (bcid >= 0) & (bcid < section_max)
                bseg = (bz[bok, 0] * section_max + bcid[bok]) * 2   # nuc slot
                ecnt = torch.zeros(G, device=dev).index_add_(
                    0, bseg, torch.ones(bseg.shape[0], device=dev))
            else:
                bz = None
                bok = torch.zeros(0, dtype=torch.bool, device=dev)
                ecnt = torch.zeros(G, device=dev)

            # ---- exact integer median via per-segment histogram ----
            # FIX: bin PER CHANNEL so B is that channel's range, not the whole
            # tile's. Prevents G*B blowing up on wide-dynamic-range 16-bit data.
            med = torch.full((G, C), float('nan'), device=dev)
            for ci in range(C):
                vmin_c = vals[:, ci].min().long()
                vmax_c = vals[:, ci].max().long()
                Bc = int((vmax_c - vmin_c).item()) + 1
                vi_c = (vals[:, ci].long() - vmin_c)         # (N,) bin index
                hist = torch.zeros(G * Bc, device=dev)
                hist.index_add_(0, seg * Bc + vi_c, ones)
                cdf = hist.view(G, Bc).cumsum(1)

                if even_median == 'lower':
                    tgt = torch.clamp((cnt + 1) // 2, min=1).unsqueeze(1)
                    med[:, ci] = (cdf >= tgt).float().argmax(1).float() + vmin_c
                else:  # 'average', matches numpy
                    k_lo = torch.clamp((cnt + 1) // 2, min=1).unsqueeze(1)
                    k_hi = torch.clamp(cnt // 2 + 1, min=1).unsqueeze(1)
                    lo = (cdf >= k_lo).float().argmax(1).float() + vmin_c
                    hi = (cdf >= k_hi).float().argmax(1).float() + vmin_c
                    med[:, ci] = (lo + hi) / 2

            # ---- derived intensity stats (G-space) ----
            n = cnt.clamp(min=1).unsqueeze(1)
            mean = (s1 / n).float()
            var  = torch.clamp((s2 / n).float() - mean * mean, min=0)
            std  = var.sqrt()
            s1f  = s1.float()

            empty = (cnt == 0)
            # FIX: NaN the SUM on empty too, so NUC_SUM_FRAC isn't polluted.
            mean[empty], std[empty], med[empty] = float('nan'), float('nan'), float('nan')
            vmin_t[empty], vmax_t[empty] = float('nan'), float('nan')
            s1f[empty] = float('nan')

            g = torch.arange(G, device=dev)
            gt   = g // (section_max * 2)
            gc   = (g // 2) % section_max
            gnuc = (g % 2) == 0

            def put(slot_mask, metrics):
                sel = g[slot_mask]
                tt, cc = gt[sel], gc[sel]
                for m_i, tensor in metrics:
                    out[tt, :, sidx, cc, m_i] = tensor[sel]

            put(gnuc, [(M.NUC_MEAN, mean), (M.NUC_MEDIAN, med), (M.NUC_SUM, s1f),
                       (M.NUC_STD, std), (M.NUC_MIN, vmin_t), (M.NUC_MAX, vmax_t),
                       (M.NUC_N, cnt.unsqueeze(1).expand(-1, C))])
            put(~gnuc, [(M.CYTO_MEAN, mean), (M.CYTO_MEDIAN, med), (M.CYTO_SUM, s1f),
                        (M.CYTO_STD, std), (M.CYTO_MIN, vmin_t), (M.CYTO_MAX, vmax_t),
                        (M.CYTO_N, cnt.unsqueeze(1).expand(-1, C))])

            # ratios (14-16), indexed off the nuclear slot
            nuc_g, cyto_g = g[gnuc], g[~gnuc]
            tt, cc = gt[nuc_g], gc[nuc_g]
            nm, cm = mean[nuc_g], mean[cyto_g]
            ns, cs_ = s1f[nuc_g], s1f[cyto_g]
            denom = ns + cs_
            out[tt, :, sidx, cc, M.EDGE_N]        = ecnt[nuc_g].unsqueeze(1).expand(-1, C)
            out[tt, :, sidx, cc, M.NC_MEAN_RATIO] = nm / cm
            # guard: NaN where there is no cyto (denom == ns and cs_ NaN handled)
            out[tt, :, sidx, cc, M.NUC_SUM_FRAC]  = ns / denom

            # ===================================================================
            # G2-space segments (t, cell) : geometry + multichannel features
            # ===================================================================
            cseg = t_i * section_max + cid                 # (N,) cell-level
            G2 = T * section_max
            nuc_m = (is_cyto == 0)
            cyt_m = ~nuc_m

            # --- nuclear second moments (local coords, one fused scatter) ---
            xi = i_i[nuc_m].float()
            yj = j_i[nuc_m].float()
            csn = cseg[nuc_m]
            vn  = vals[nuc_m]                               # (Nn, C)

            feat = torch.stack([torch.ones_like(xi), xi, yj,
                                xi * xi, yj * yj, xi * yj], dim=1)
            acc = torch.zeros(G2, 6, device=dev).index_add_(0, csn, feat)
            Nn, Sx, Sy, Sxx, Syy, Sxy = acc.unbind(1)
            nn = Nn.clamp(min=1)

            mx, my = Sx / nn, Sy / nn
            Cxx = Sxx / nn - mx * mx
            Cyy = Syy / nn - my * my
            Cxy = Sxy / nn - mx * my

            mid = 0.5 * (Cxx + Cyy)
            rng = torch.sqrt(torch.clamp(((Cxx - Cyy) * 0.5) ** 2 + Cxy * Cxy, min=0))
            lam1 = mid + rng                               # major variance
            lam2 = (mid - rng).clamp(min=0)                # minor variance

            eccentricity = torch.sqrt((1 - lam2 / lam1.clamp(min=1e-6)).clamp(0, 1))
            elongation   = torch.sqrt(lam1.clamp(min=1e-6) / lam2.clamp(min=1e-6))
            orientation  = 0.5 * torch.atan2(2 * Cxy, Cxx - Cyy)
            major_len    = 4.0 * torch.sqrt(lam1)
            minor_len    = 4.0 * torch.sqrt(lam2)
            ellipse_area = math.pi * (major_len / 2) * (minor_len / 2)
            ellipse_ratio = Nn / ellipse_area.clamp(min=1e-6)
            r_nuc = torch.sqrt(Nn.clamp(min=1) / math.pi)  # equiv-circle radius

            # --- Hu moment invariants 1 & 2 (normalized central moments) ---
            # eta_pq = mu_pq / mu_00^(1 + (p+q)/2); for p+q=2, exponent = 2.
            # mu20/mu02/mu11 are the *unnormalized* central moments = C** * Nn.
            mu20 = Cxx * Nn
            mu02 = Cyy * Nn
            mu11 = Cxy * Nn
            norm = Nn.clamp(min=1) ** 2
            eta20, eta02, eta11 = mu20 / norm, mu02 / norm, mu11 / norm
            hu1 = eta20 + eta02
            hu2 = (eta20 - eta02) ** 2 + 4 * eta11 ** 2

            # global nuclear centroid
            ncx = Sx / nn + lx
            ncy = Sy / nn + ly

            # --- bounding box: extent + aspect ---
            POS = torch.full((G2,), 1e9, device=dev)
            NEG = torch.full((G2,), -1e9, device=dev)
            imin = POS.clone().index_reduce_(0, csn, xi, 'amin', include_self=True)
            imax = NEG.clone().index_reduce_(0, csn, xi, 'amax', include_self=True)
            jmin = POS.clone().index_reduce_(0, csn, yj, 'amin', include_self=True)
            jmax = NEG.clone().index_reduce_(0, csn, yj, 'amax', include_self=True)
            h = (imax - imin + 1).clamp(min=1)
            w = (jmax - jmin + 1).clamp(min=1)
            extent      = Nn / (h * w)
            bbox_aspect = h / w

            # --- per-channel power sums on nuclear pixels (fused, float64) ---
            pw = torch.stack([vn, vn ** 2, vn ** 3, vn ** 4], dim=2).double()
            pacc = torch.zeros(G2, C, 4, device=dev, dtype=torch.float64).index_add_(
                0, csn, pw)
            p1, p2, p3, p4 = (t.float() for t in pacc.unbind(2))

            Nnc = Nn.clamp(min=1).unsqueeze(1)
            mu_n = p1 / Nnc
            m2 = (p2 / Nnc - mu_n ** 2).clamp(min=1e-8)
            m3 = p3 / Nnc - 3 * mu_n * (p2 / Nnc) + 2 * mu_n ** 3
            m4 = (p4 / Nnc - 4 * mu_n * (p3 / Nnc)
                  + 6 * mu_n ** 2 * (p2 / Nnc) - 3 * mu_n ** 4)
            g1 = m3 / m2 ** 1.5
            g2_ = m4 / m2 ** 2 - 3
            nk = Nn.clamp(min=4).unsqueeze(1)
            denom_b = g2_ + 3 * (nk - 1) ** 2 / ((nk - 2) * (nk - 3))
            # FIX: clamp denom and require n>=8 for a meaningful bimodality est.
            bimodality = (g1 ** 2 + 1) / denom_b.clamp(min=1e-3)
            bimodality[Nn < 8] = float('nan')

            # --- per-channel intensity-weighted centroid offset ---
            vw = vn.clamp(min=0)
            gx = (i_i[nuc_m] + lx).float().unsqueeze(1)
            gy = (j_i[nuc_m] + ly).float().unsqueeze(1)
            wsum = torch.zeros(G2, C, device=dev).index_add_(0, csn, vw)
            wx   = torch.zeros(G2, C, device=dev).index_add_(0, csn, vw * gx)
            wy   = torch.zeros(G2, C, device=dev).index_add_(0, csn, vw * gy)
            wden = wsum.clamp(min=1e-6)
            dcx = wx / wden - ncx.unsqueeze(1)
            dcy = wy / wden - ncy.unsqueeze(1)
            chan_offset = torch.sqrt(dcx * dcx + dcy * dcy) / r_nuc.clamp(min=1e-6).unsqueeze(1)

            # --- within-nucleus cross-channel Pearson correlations ---
            sd_n = m2.sqrt()
            corr = torch.full((G2, len(_CORR_PAIRS)), float('nan'), device=dev)
            for pi, (a, b) in enumerate(_CORR_PAIRS):
                if a < C and b < C:
                    sab = torch.zeros(G2, device=dev).index_add_(
                        0, csn, vn[:, a] * vn[:, b])
                    cov = sab / Nn.clamp(min=1) - mu_n[:, a] * mu_n[:, b]
                    corr[:, pi] = cov / (sd_n[:, a] * sd_n[:, b]).clamp(min=1e-8)

            # --- radial roughness + circularity (from border ring) ---
            radial_rough = torch.full((G2,), float('nan'), device=dev)
            circularity  = torch.full((G2,), float('nan'), device=dev)
            if bz is not None and bok.any():
                bt2 = bz[bok, 0]
                bi2 = bz[bok, 1].float() + lx
                bj2 = bz[bok, 2].float() + ly
                bcell = bt2 * section_max + bcid[bok]
                br = torch.sqrt((bi2 - ncx[bcell]) ** 2 + (bj2 - ncy[bcell]) ** 2)
                bacc = torch.zeros(G2, 3, device=dev).index_add_(
                    0, bcell, torch.stack([torch.ones_like(br), br, br * br], 1))
                be, bs, bsq = bacc.unbind(1)
                be1 = be.clamp(min=1)
                rmean_b = bs / be1
                rstd_b = (bsq / be1 - rmean_b ** 2).clamp(min=0).sqrt()
                radial_rough = rstd_b / rmean_b.clamp(min=1e-6)
                circularity  = 4 * math.pi * Nn / be.clamp(min=1) ** 2

            # --- concentricity: nuclear vs cytoplasmic centroid offset ---
            xi_c = i_i[cyt_m].float()
            yj_c = j_i[cyt_m].float()
            csc  = cseg[cyt_m]
            cacc = torch.zeros(G2, 3, device=dev).index_add_(
                0, csc, torch.stack([torch.ones_like(xi_c), xi_c, yj_c], 1))
            cN, cSx, cSy = cacc.unbind(1)
            ccx = cSx / cN.clamp(min=1) + lx
            ccy = cSy / cN.clamp(min=1) + ly
            concentricity = torch.full((G2,), float('nan'), device=dev)
            both = (Nn > 0) & (cN > 0)
            concentricity[both] = (torch.sqrt((ncx[both] - ccx[both]) ** 2
                                              + (ncy[both] - ccy[both]) ** 2)
                                   / r_nuc[both].clamp(min=1e-6))

            # ---- assemble G2 metrics ----
            g2 = torch.arange(G2, device=dev)
            g2t = g2 // section_max
            g2c = g2 % section_max
            empty2 = (Nn == 0)

            cell_label = (sidx * 1000 + g2c + 1).float()

            def put2(m_i, tensor):
                v = tensor.clone()
                v[empty2] = float('nan')
                out[g2t, :, sidx, g2c, m_i] = v.unsqueeze(1).expand(-1, C)

            def put2c(m_i, tensor):
                v = tensor.clone()
                v[empty2] = float('nan')
                out[g2t, :, sidx, g2c, m_i] = v

            put2(M.CENTROID_X,        ncx)
            put2(M.CENTROID_Y,        ncy)
            put2(M.CELL_LABEL,        cell_label)
            put2(M.NUC_AREA,          Nn)
            put2(M.NUC_ECCENTRICITY,  eccentricity)
            put2(M.NUC_ELONGATION,    elongation)
            put2(M.NUC_ORIENTATION,   orientation)
            put2(M.NUC_MAJOR_LEN,     major_len)
            put2(M.NUC_MINOR_LEN,     minor_len)
            put2(M.NUC_ELLIPSE_RATIO, ellipse_ratio)
            put2(M.NUC_EXTENT,        extent)
            put2(M.NUC_BBOX_ASPECT,   bbox_aspect)
            put2(M.NUC_RADIAL_ROUGH,  radial_rough)
            put2(M.NUC_CIRCULARITY,   circularity)
            put2(M.NUC_HU1,           hu1)
            put2(M.NUC_HU2,           hu2)
            put2(M.CONCENTRICITY,     concentricity)

            put2c(M.CHAN_CENTROID_OFF, chan_offset)
            put2c(M.NUC_BIMODALITY,    bimodality)

            for slot, pi in zip(_CORR_SLOTS, range(len(_CORR_PAIRS))):
                put2(slot, corr[:, pi])

    # =======================================================================
    # POST-STEP: local density (crowding). Computed on the flat centroid list
    # per frame, O(k^2) over the few thousand cells per frame -- cheap, and it
    # doesn't fit the per-subsection pixel loop because neighbors can straddle
    # subsection borders.
    # =======================================================================
    if density_post:
        _fill_local_density(out, subsections, section_max, T, C)

    return out


def _fill_local_density(out, subsections, section_max, T, C):
    """For each nucleus, count other nuclei whose centroid lies within 2*r_nuc.
    Uses CENTROID_X/Y and NUC_AREA already written into `out`."""
    dev = out.device
    S2 = subsections ** 2
    cx = out[:, 0, :, :, M.CENTROID_X].reshape(T, -1)   # (T, S2*section_max)
    cy = out[:, 0, :, :, M.CENTROID_Y].reshape(T, -1)
    area = out[:, 0, :, :, M.NUC_AREA].reshape(T, -1)
    r = torch.sqrt(area.clamp(min=1) / math.pi)
    for t in range(T):
        m = ~torch.isnan(cx[t])
        if m.sum() < 2:
            continue
        idx = m.nonzero(as_tuple=True)[0]
        px, py, pr = cx[t, idx], cy[t, idx], r[t, idx]
        d = torch.cdist(torch.stack([px, py], 1), torch.stack([px, py], 1))
        thr = (2 * pr).unsqueeze(1)
        within = (d <= thr) & (d > 0)
        dens = within.sum(1).float()
        flat = out[t, :, :, :, M.LOCAL_DENSITY].reshape(C, -1)
        flat[:, idx] = dens.unsqueeze(0).expand(C, -1)
        out[t, :, :, :, M.LOCAL_DENSITY] = flat.reshape(C, S2, section_max)


# ---------------------------------------------------------------------- packaging
# The saved metric names, each mapped to its slot in the measurement tensor.
_SCALAR_SLOTS = {
    "x": M.CENTROID_X, "y": M.CENTROID_Y, "label": M.CELL_LABEL, "nuc_area": M.NUC_AREA, "cyto_area": M.CYTO_N,
    "edge_n": M.EDGE_N, "eccentricity": M.NUC_ECCENTRICITY, "elongation": M.NUC_ELONGATION,
    "orientation": M.NUC_ORIENTATION, "major_len": M.NUC_MAJOR_LEN, "minor_len": M.NUC_MINOR_LEN,
    "ellipse_ratio": M.NUC_ELLIPSE_RATIO, "extent": M.NUC_EXTENT, "bbox_aspect": M.NUC_BBOX_ASPECT,
    "radial_rough": M.NUC_RADIAL_ROUGH, "circularity": M.NUC_CIRCULARITY, "hu1": M.NUC_HU1, "hu2": M.NUC_HU2,
    "concentricity": M.CONCENTRICITY, "corr_01": M.CORR_01, "corr_02": M.CORR_02, "corr_12": M.CORR_12,
    "local_density": M.LOCAL_DENSITY,
}
_CHANNEL_SLOTS = {
    "nuc_mean": M.NUC_MEAN, "nuc_median": M.NUC_MEDIAN, "nuc_sum": M.NUC_SUM, "nuc_std": M.NUC_STD,
    "nuc_min": M.NUC_MIN, "nuc_max": M.NUC_MAX, "cyto_mean": M.CYTO_MEAN, "cyto_median": M.CYTO_MEDIAN,
    "cyto_sum": M.CYTO_SUM, "cyto_std": M.CYTO_STD, "cyto_min": M.CYTO_MIN, "cyto_max": M.CYTO_MAX,
    "nc_ratio": M.NC_MEAN_RATIO, "nuc_sum_frac": M.NUC_SUM_FRAC, "chan_centroid_off": M.CHAN_CENTROID_OFF,
    "bimodality": M.NUC_BIMODALITY,
}
SCALARS = ["t", *_SCALAR_SLOTS, "underseg_score", "touches_border", "cyto_kept"]
CHANNELS = list(_CHANNEL_SLOTS)
S = {name: i for i, name in enumerate(SCALARS)}
K = {name: i for i, name in enumerate(CHANNELS)}


def flatten_measurements(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the slots that hold a cell. Rows are ordered by frame, then label."""
    t, s, c = ((out[:, 0, :, :, M.NUC_N] > 0)).nonzero(as_tuple=True)
    device = out.device
    scalars = torch.full((t.numel(), len(SCALARS)), torch.nan, device=device)
    scalars[:, S["t"]] = t.float()
    for name, slot in _SCALAR_SLOTS.items():
        scalars[:, S[name]] = out[t, 0, s, c, slot]
    channels = torch.stack([out[t, :, s, c, slot] for slot in _CHANNEL_SLOTS.values()], dim=-1)
    return scalars, channels


def mark_border_cells(scalars: torch.Tensor, label_map: torch.Tensor) -> None:
    """Flag nuclei touching the field edge: their area and shape are underestimates."""
    T = label_map.shape[0]
    edges = torch.cat([label_map[:, 0, 0, :], label_map[:, 0, -1, :], label_map[:, 0, :, 0], label_map[:, 0, :, -1]], dim=1)
    frames = torch.arange(T, device=label_map.device)[:, None].expand_as(edges)
    positive = edges > 0
    span = int(scalars[:, S["label"]].max().item()) + 1 if scalars.shape[0] else 1
    touched = torch.unique(frames[positive].long() * span + edges[positive].long())
    keys = scalars[:, S["t"]].long() * span + scalars[:, S["label"]].long()
    scalars[:, S["touches_border"]] = torch.isin(keys, touched).float()


def _ring_counts(scalars: torch.Tensor, ring_labels: torch.Tensor) -> torch.Tensor:
    """Pixels in each cell's cytoplasm ring before the significance mask, one per row of `scalars`."""
    T = ring_labels.shape[0]
    span = int(max(scalars[:, S["label"]].max().item(), ring_labels.max().item())) + 1
    frames = torch.arange(T, device=ring_labels.device).view(T, 1, 1, 1).expand_as(ring_labels)
    on = ring_labels > 0
    counts = torch.bincount(frames[on].long() * span + ring_labels[on].long(), minlength=T * span)
    keys = scalars[:, S["t"]].long() * span + scalars[:, S["label"]].long()
    return counts[keys].float()


_CYTO_CHANNELS = [K[n] for n in ("cyto_mean", "cyto_median", "cyto_sum", "cyto_std", "cyto_min", "cyto_max", "nc_ratio", "nuc_sum_frac")]


def apply_min_ring(scalars: torch.Tensor, channels: torch.Tensor, ring_labels: torch.Tensor | None, min_pixels: int,
                   min_kept: float = 0.0) -> None:
    """Fill `cyto_kept` (kept ring pixels / ring pixels; 1 without a mask) and blank the cytoplasm
    measurements of cells with fewer than `min_pixels` ring pixels left, or less than `min_kept` of their
    ring: NaN, not a ratio of a few pixels. With the significance mask, a cell that keeps little of its
    ring is nearly always one that does not express the mask channel's reporter."""
    kept = scalars[:, S["cyto_area"]].nan_to_num(0)
    if ring_labels is None:
        scalars[:, S["cyto_kept"]] = torch.where(kept > 0, 1.0, torch.nan)
    else:
        total = _ring_counts(scalars, ring_labels)
        scalars[:, S["cyto_kept"]] = torch.where(total > 0, kept / total.clamp(min=1), torch.nan)
    few = kept < max(int(min_pixels), 1)
    if ring_labels is not None and min_kept > 0:
        few |= ~(scalars[:, S["cyto_kept"]] >= min_kept)
    if few.any():
        rows = channels[few]
        rows[:, :, _CYTO_CHANNELS] = torch.nan
        channels[few] = rows
        scalars[few, S["concentricity"]] = torch.nan


def measure_batch(corrected: torch.Tensor, label_map: torch.Tensor, border_labels: torch.Tensor, section_max: int,
                  radius: int, cfg: dict, ring_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure a batch and return flat tensors. Uses int32 data: int16 wraps above 32767.

    ring_labels: every cytoplasm-ring pixel's label before the significance mask (see labels.mask_ring),
    or None when the ring was not masked."""
    device = corrected.device
    C = corrected.shape[1]
    if section_max == 0:
        return torch.zeros(0, len(SCALARS), device=device), torch.zeros(0, C, len(CHANNELS), device=device)
    p = cfg["measure"]
    data = corrected.round().to(torch.int32)
    out = measure_subsections(data, label_map, border_labels, section_max, radius, subsections=int(p["subsections"]), even_median=p["even_median"])
    scalars, channels = flatten_measurements(out)
    mark_border_cells(scalars, label_map)
    apply_min_ring(scalars, channels, ring_labels, int(p.get("cyto_min_pixels", 1)), float(p.get("cyto_min_kept", 0.0)))
    return scalars, channels
