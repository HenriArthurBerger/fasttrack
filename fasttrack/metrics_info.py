"""Plain names, descriptions and formulas for every measurement, as computed in track/measure.py.

Shown in the Analysis tab under the metric choice. "Nucleus" is the segmented nucleus; "ring" is the
cytoplasm ring, a band two pixels wide just outside the nucleus (one pixel gap). Intensities are
background-corrected counts. Lengths are in pixels.
"""
from __future__ import annotations

# name -> (plain name, what it tells you, formula as computed)
INFO: dict[str, tuple[str, str, str]] = {
    # ------------------------------------------------------------ per channel
    "nuc_mean": ("Nuclear mean intensity", "Average brightness of the nucleus in this channel.",
                 "Σ I(p) / N over nuclear pixels p; N = nuclear area"),
    "nuc_median": ("Nuclear median intensity", "Typical brightness of the nucleus, robust to a few very bright or dark pixels.",
                   "median of I(p) over nuclear pixels"),
    "nuc_sum": ("Nuclear total intensity", "Total signal in the nucleus: tracks DNA content with H2B, total protein amount otherwise.",
                "Σ I(p) over nuclear pixels  (= mean × area)"),
    "nuc_std": ("Nuclear intensity spread", "How uneven the signal is inside the nucleus (speckles, foci, chromatin texture).",
                "standard deviation of I(p) over nuclear pixels"),
    "nuc_min": ("Nuclear minimum intensity", "Darkest pixel in the nucleus.", "min I(p) over nuclear pixels"),
    "nuc_max": ("Nuclear maximum intensity", "Brightest pixel in the nucleus.", "max I(p) over nuclear pixels"),
    "cyto_mean": ("Ring mean intensity", "Average brightness just outside the nucleus: a proxy for the cytoplasm next to it.",
                  "Σ I(p) / N_ring over ring pixels"),
    "cyto_median": ("Ring median intensity", "Typical brightness just outside the nucleus. KTR readouts use this over the nuclear median.",
                    "median of I(p) over ring pixels"),
    "cyto_sum": ("Ring total intensity", "Total signal in the ring.", "Σ I(p) over ring pixels"),
    "cyto_std": ("Ring intensity spread", "How uneven the ring signal is.", "standard deviation of I(p) over ring pixels"),
    "cyto_min": ("Ring minimum intensity", "Darkest ring pixel.", "min I(p) over ring pixels"),
    "cyto_max": ("Ring maximum intensity", "Brightest ring pixel.", "max I(p) over ring pixels"),
    "nc_ratio": ("Nucleus / ring ratio", "How much more the signal is in the nucleus than around it (nuclear translocation).",
                 "nuc_mean / cyto_mean  (no background floor; use a ratio metric with a background percentile for KTRs)"),
    "nuc_sum_frac": ("Nuclear fraction of signal", "Share of the nucleus-plus-ring signal that is in the nucleus, 0 to 1.",
                     "nuc_sum / (nuc_sum + cyto_sum)"),
    "chan_centroid_off": ("Signal off-centre", "How far the channel's bright centre sits from the nucleus centre, in nuclear radii. "
                          "High for signal on one side, or two nuclei counted as one.",
                          "|intensity-weighted centroid − nuclear centroid| / r;  r = √(area / π)"),
    "bimodality": ("Bimodality of nuclear pixels", "Whether nuclear pixel intensities form two groups (e.g. two merged nuclei of "
                   "different brightness). Above ~0.555 suggests two modes. Needs 8+ pixels.",
                   "Sarle's b = (g1² + 1) / (g2 + 3(n−1)² / ((n−2)(n−3)));  g1 = skewness, g2 = excess kurtosis"),
    # ------------------------------------------------------------ per cell
    "nuc_area": ("Nuclear area", "Size of the nucleus in pixels. Rises through the cell cycle, drops at division.",
                 "N = number of nuclear pixels"),
    "cyto_area": ("Ring area", "Pixels in the cytoplasm ring that were measured (smaller where neighbours crowd it, or where the "
                  "significance mask drops pixels that are not above background).", "number of ring pixels kept"),
    "cyto_kept": ("Ring pixels kept", "Fraction of the cytoplasm ring that passed the significance mask (1 without a mask). Low "
                  "where the ring runs off the cell. If it tracks the readout, the mask is biasing it.",
                  "kept ring pixels / ring pixels;  kept: corrected > background level + k·σ (mask channel)"),
    "edge_n": ("Nuclear perimeter", "Length of the nuclear boundary, in boundary pixels.", "number of pixels on the nuclear border"),
    "eccentricity": ("Eccentricity", "0 for a circle, towards 1 for a long thin shape.",
                     "√(1 − λ2 / λ1);  λ1 ≥ λ2 = eigenvalues of the pixel-coordinate covariance"),
    "elongation": ("Elongation", "Long axis over short axis; 1 for a circle.", "√(λ1 / λ2)"),
    "orientation": ("Orientation", "Angle of the long axis, in radians (−π/2 to π/2), from the first image axis.",
                    "½ atan2(2 C_xy, C_xx − C_yy)"),
    "major_len": ("Long axis length", "Length of the nucleus along its long axis (equivalent ellipse).", "4 √λ1"),
    "minor_len": ("Short axis length", "Width of the nucleus across its short axis (equivalent ellipse).", "4 √λ2"),
    "ellipse_ratio": ("Ellipse fill", "How well an ellipse describes the shape; 1 = perfect ellipse, lower = irregular or lobed.",
                      "area / (π · major/2 · minor/2)"),
    "extent": ("Bounding box fill", "Share of the nucleus's bounding box it fills.", "area / (box height × box width)"),
    "bbox_aspect": ("Bounding box aspect", "Bounding box height over width (depends on orientation).", "box height / box width"),
    "radial_rough": ("Boundary roughness", "How much the distance from centre to edge varies around the nucleus; 0 for a circle.",
                     "std(r_edge) / mean(r_edge);  r_edge = distance of each border pixel to the centroid"),
    "circularity": ("Circularity", "Compactness of the outline; about 1 for a circle, lower for irregular or merged shapes.",
                    "4π · area / perimeter²"),
    "hu1": ("Hu moment 1", "Spread of the shape about its centre, independent of size, position and rotation.",
            "η20 + η02;  η_pq = μ_pq / μ00²  (central moments)"),
    "hu2": ("Hu moment 2", "How elongated the shape is, independent of size, position and rotation.", "(η20 − η02)² + 4 η11²"),
    "concentricity": ("Nucleus off-centre in its ring", "Distance between the nucleus centre and its ring's centre, in nuclear radii. "
                      "High when neighbours eat one side of the ring.", "|nuclear centroid − ring centroid| / r;  r = √(area / π)"),
    "corr_01": ("Correlation, channels 1 and 2", "How well the two channels' brightness patterns match inside the nucleus (−1 to 1).",
                "Pearson r of I₁(p) and I₂(p) over nuclear pixels"),
    "corr_02": ("Correlation, channels 1 and 3", "How well the two channels' brightness patterns match inside the nucleus (−1 to 1).",
                "Pearson r of I₁(p) and I₃(p) over nuclear pixels"),
    "corr_12": ("Correlation, channels 2 and 3", "How well the two channels' brightness patterns match inside the nucleus (−1 to 1).",
                "Pearson r of I₂(p) and I₃(p) over nuclear pixels"),
    "local_density": ("Neighbours nearby", "Number of other nuclei whose centre lies within two nuclear radii of this one.",
                      "#{ j ≠ i : |c_j − c_i| ≤ 2 r_i },  r_i = √(area_i / π)"),
    "underseg_score": ("Merged-nuclei score", "Likelihood (0-100) that this object is two nuclei segmented as one, from its shape "
                       "and channel features. Calibrated on synthetic data.", "logistic model of shape and intensity features"),
    "touches_border": ("Touches the image border", "1 if the nucleus touches the edge of the field (cut off), else 0.", "0 or 1"),
    "speed": ("Speed", "Distance the nucleus moved since the track's previous frame, in pixels per frame, with stage drift "
              "removed. Use a moving mean to smooth out jitter of the centre.",
              "|c(t) − c(t')| / (t − t'),  c = centroid + cumulative stage shift, t' = the track's previous observed frame"),
    "x": ("Position x", "Centre along the first image axis, in pixels.", "Σ x / N over nuclear pixels"),
    "y": ("Position y", "Centre along the second image axis, in pixels.", "Σ y / N over nuclear pixels"),
}


def label(name: str) -> str:
    return INFO.get(name, (name, "", ""))[0]


def describe(name: str) -> str:
    """'Plain name (code name)' — description — formula, as rich text."""
    plain, what, formula = INFO.get(name, (name, "", ""))
    return f"<b>{plain}</b> <span style='color:#777'>({name})</span><br>{what}<br><span style='color:#555'><i>{formula}</i></span>"
