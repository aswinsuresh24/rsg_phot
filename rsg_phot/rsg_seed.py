import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from numpy.polynomial import polynomial as P
from scipy.stats import median_abs_deviation

def fit_gmm_bic(colors, n_min=3, n_max=6, n_init=5, random_state=42):
    """
    Fit GMMs with n_min..n_max components to a 1-D colour array.
    Returns the model with the lowest BIC, together with the BIC curve.

    Parameters
    ----------
    colors        : (N,) array of colour values for stars in one magnitude slice
    n_min, n_max  : range of component counts to try
    n_init        : number of random initialisations per fit (guards against
                    local minima)
    random_state  : RNG seed for reproducibility

    Returns
    -------
    best_gmm : fitted GaussianMixture with the lowest BIC
    bics     : list of BIC values for n_min..n_max
    n_range  : list of n values tried
    """
    bics, models = [], []
    n_range = list(range(n_min, n_max + 1))

    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(colors.reshape(-1, 1))
        bics.append(gmm.bic(colors.reshape(-1, 1)))
        models.append(gmm)

    best_idx = int(np.argmin(bics))
    return models[best_idx], bics, n_range


def slice_and_fit(mag, color, mag_bins, n_max=5, min_stars=30, n_init=5):
    """
    Slice the CMD into magnitude bins and fit a BIC-optimal GMM per slice.

    Parameters
    ----------
    mag, color : (N,) arrays
    mag_bins   : bin edges (e.g. np.arange(mag.min(), mag.max(), 0.3))
    n_max      : maximum number of GMM components to try
    min_stars  : slices with fewer stars are skipped
    n_init     : passed to fit_gmm_bic

    Returns
    -------
    results : dict  {(m_lo, m_hi): slice_result | None}
        Each slice_result contains:
            gmm     - fitted GaussianMixture
            bics    - BIC curve
            n_range - n values tried
            n_best  - best fit number of components
            means   - component means sorted blue-red
            sigmas  - component standard deviations (same order)
            weights - component weights (same order)
            colors  - colour values of stars in this slice
            mag_mid - midpoint of the magnitude bin
    """
    results = {}

    for i in range(len(mag_bins) - 1):
        m_lo, m_hi = mag_bins[i], mag_bins[i + 1]
        mask = (mag >= m_lo) & (mag < m_hi)
        c = color[mask]

        if len(c) < min_stars:
            results[(m_lo, m_hi)] = None
            continue

        best_gmm, bics, n_range = fit_gmm_bic(c, n_max=n_max, n_init=n_init)

        order   = np.argsort(best_gmm.means_.flatten())
        means   = best_gmm.means_.flatten()[order]
        sigmas  = np.sqrt(best_gmm.covariances_.flatten())[order]
        weights = best_gmm.weights_[order]

        results[(m_lo, m_hi)] = {
            "gmm":     best_gmm,
            "bics":    bics,
            "n_range": n_range,
            "n_best":  best_gmm.n_components,
            "means":   means,
            "sigmas":  sigmas,
            "weights": weights,
            "colors":  c,
            "mag_mid": np.median(mag[mask]),
        }

    return results


def initialize_rsg_chain(results, min_weight=0.05, exp_rsg_color=0.5):
    """
    Identify the RSG candidate component in each slice based on the expected
    color of the RSG branch

    Parameters
    ----------
    results    : output of slice_and_fit
    min_weight : components with weight below this are ignored

    Returns
    -------
    candidates : dict  {(m_lo, m_hi): candidate_dict | None}
        Each candidate_dict contains mean, sigma, weight, gmm_idx (sorted),
        is_reddest flag, and mag_mid.
    """
    candidates = {}

    for key, res in results.items():
        if res is None:
            candidates[key] = None
            continue

        # Filter negligible components
        ok = np.array(res["weights"]) >= min_weight
        means   = np.array(res["means"])[ok]
        sigmas  = np.array(res["sigmas"])[ok]
        weights = np.array(res["weights"])[ok]
        # Keep track of original sorted indices so we can map back to GMM
        orig_idx = np.where(ok)[0]

        if len(means) < 2:
            candidates[key] = None
            continue

        rsg_local_idx    = np.argmin(np.abs(means - exp_rsg_color))  # closest to expected RSG color
        rsg_sorted_idx   = int(orig_idx[rsg_local_idx]) # index in sorted order

        candidates[key] = {
            "mean":       float(means[rsg_local_idx]),
            "sigma":      float(sigmas[rsg_local_idx]),
            "weight":     float(weights[rsg_local_idx]),
            "gmm_idx":    rsg_sorted_idx,
            "is_reddest": rsg_sorted_idx == len(res["means"]) - 1,
            "mag_mid":    res["mag_mid"],
            "key":        key,
        }

    return candidates


def fit_and_validate_chain(candidates, deg=2, sigma_clip=2.0):
    """
    Fit a smooth polynomial (colour vs magnitude) through the per-slice RSG
    candidates and sigma-clip outliers.  Outlier slices are typically ones
    where the GMM failed (merged blue+RSG or split AGB into RSG region).

    Parameters
    ----------
    candidates : output of initialize_chain_by_gap
    deg        : polynomial degree (1 = linear, 2 = quadratic)
    sigma_clip : rejection threshold in units of MAD-based sigma

    Returns
    -------
    validated : dict with same keys as candidates; each entry gains
                  on_trend  - bool
                  predicted - colour predicted by the trend at this magnitude
                  residual  - observed minus predicted
    trend_fn  : callable  mag -> predicted_colour
    """
    keys  = [k for k, v in candidates.items() if v is not None]
    mags  = np.array([candidates[k]["mag_mid"] for k in keys])
    means = np.array([candidates[k]["mean"]    for k in keys])

    mask = np.ones(len(mags), dtype=bool)

    for _ in range(10):
        if mask.sum() < deg + 2:
            break
        coeffs    = P.polyfit(mags[mask], means[mask], deg=deg)
        predicted = P.polyval(mags, coeffs)
        residuals = means - predicted
        mad       = median_abs_deviation(residuals[mask])
        mask      = np.abs(residuals) < sigma_clip * mad * 1.4826

    coeffs    = P.polyfit(mags[mask], means[mask], deg=deg)
    predicted = P.polyval(mags, coeffs)

    validated = {}
    for i, k in enumerate(keys):
        c = candidates[k].copy()
        c["on_trend"]  = bool(mask[i])
        c["predicted"] = float(predicted[i])
        c["residual"]  = float(means[i] - predicted[i])
        validated[k]   = c

    # Fill entries that were None in candidates
    for k, v in candidates.items():
        if k not in validated:
            validated[k] = None

    trend_fn = lambda m: float(P.polyval(np.asarray(m, dtype=float), coeffs))

    return validated, trend_fn


def recover_merged_slices(validated, results, trend_fn,
                           max_residual=0.12, min_weight=0.02):
    """
    For slices where the RSG candidate deviates from the smooth colour trend
    (likely because the GMM merged the blue/foreground population with RSGs,
    pulling the component mean blueward), attempt to substitute the component
    whose mean is closest to the trend prediction.

    If no suitable component exists, fall back to the trend colour itself
    (flagged as interpolated=True); those stars will receive low posterior
    probabilities and will be down-weighted at the extraction step.

    Parameters
    ----------
    validated     : output of fit_and_validate_chain
    results       : output of slice_and_fit
    trend_fn      : callable from fit_and_validate_chain
    max_residual  : maximum allowed |mean - trend| for a recovery candidate
    min_weight    : minimum component weight to consider

    Returns
    -------
    validated : updated in place and returned
    """
    for key, cand in validated.items():
        if cand is None or cand["on_trend"]:
            continue

        res            = results[key]
        expected_color = trend_fn(res["mag_mid"])
        means          = np.array(res["means"])
        sigmas         = np.array(res["sigmas"])
        weights        = np.array(res["weights"])

        dists = np.abs(means - expected_color)
        best  = int(np.argmin(dists))

        if dists[best] <= max_residual and weights[best] >= min_weight:
            validated[key] = {
                "mean":        float(means[best]),
                "sigma":       float(sigmas[best]),
                "weight":      float(weights[best]),
                "gmm_idx":     best,
                "is_reddest":  best == len(means) - 1,
                "mag_mid":     res["mag_mid"],
                "key":         key,
                "on_trend":    True,
                "predicted":   expected_color,
                "residual":    float(means[best] - expected_color),
                "recovered":   True,
            }
        else:
            # No good component — mark as interpolated
            validated[key]["interpolated"]  = True
            validated[key]["mean"]          = expected_color
            validated[key]["on_trend"]      = True   # treat as usable

    return validated


def extract_rsg_seed(validated, results, mag, color, prob_threshold=0.5):
    """
    Assign stars to the RSG component in each slice using the GMM posterior
    probabilities.

    Parameters
    ----------
    validated       : output of recover_merged_slices
    results         : output of slice_and_fit
    mag, color      : original full arrays
    prob_threshold  : minimum posterior probability to include a star

    Returns
    -------
    rsg_indices : (M,) integer array — indices into mag/color
    rsg_probs   : (M,) float array  — RSG posterior probability per star
    """
    rsg_indices, rsg_probs = [], []

    for key, cand in validated.items():
        if cand is None:
            continue

        m_lo, m_hi = key
        slice_mask  = np.where((mag >= m_lo) & (mag < m_hi))[0]
        c           = color[slice_mask]

        if len(c) == 0:
            continue

        gmm   = results[key]["gmm"]
        order = np.argsort(gmm.means_.flatten())

        # Map sorted index → GMM internal index
        gmm_internal_idx = int(order[cand["gmm_idx"]])

        posteriors = gmm.predict_proba(c.reshape(-1, 1))
        rsg_post   = posteriors[:, gmm_internal_idx]

        above = rsg_post >= prob_threshold
        rsg_indices.extend(slice_mask[above].tolist())
        rsg_probs.extend(rsg_post[above].tolist())

    return np.array(rsg_indices, dtype=int), np.array(rsg_probs)

def extract_seeds(validated, results, mag, color, lcut, prob_threshold=0.5):
    """
    Assign stars to RSG, AGB, or blue-star populations in each magnitude slice
    using GMM posterior probabilities.
 
    The RSG component is identified by the validated chain.  All components
    redder than the RSG component (higher sorted index) are pooled as AGBs;
    all components bluer (lower sorted index) are pooled as blue/foreground
    stars.  Within each population a star is included when the summed
    posterior probability across its assigned components meets prob_threshold.
 
    Parameters
    ----------
    validated       : output of recover_merged_slices
    results         : output of slice_and_fit
    mag, color      : original full arrays
    prob_threshold  : minimum summed posterior probability to include a star
                      in any population seed
 
    Returns
    -------
    rsg_indices  : (M,) int array   — indices of RSG seed stars
    agb_indices  : (M,) int array   — indices of AGB seed stars
    blue_indices : (M,) int array   — indices of blue/foreground seed stars
    rsg_probs    : (M,) float array — RSG posterior probability per RSG star
    agb_probs    : (M,) float array — summed AGB posterior per AGB star
    blue_probs   : (M,) float array — summed blue posterior per blue star
    """
    rsg_indices,  rsg_probs  = [], []
    agb_indices,  agb_probs  = [], []
    blue_indices, blue_probs = [], []
 
    for key, cand in validated.items():
        if cand is None:
            continue
 
        m_lo, m_hi = key
        slice_mask  = np.where((mag >= m_lo) & (mag < m_hi))[0]
        c           = color[slice_mask]
        m_          = mag[slice_mask]
        rsg_mean, rsg_sig = cand["mean"], cand["sigma"]
 
        if len(c) == 0:
            continue
 
        res   = results[key]
        gmm   = res["gmm"]
        n_comp = gmm.n_components
 
        # sorted order: index 0 = bluest component, index n-1 = reddest
        order            = np.argsort(gmm.means_.flatten())
        rsg_sorted_idx   = cand["gmm_idx"]          # position in sorted order
        rsg_internal_idx = int(order[rsg_sorted_idx])
 
        # Sorted indices for AGB (redder) and blue (bluer) components
        agb_sorted_idxs  = list(range(rsg_sorted_idx + 1, n_comp))
        if rsg_sorted_idx == 0: 
            blue_comp = 1
        else:
            blue_comp = rsg_sorted_idx
        blue_sorted_idxs = list(range(0, blue_comp))
 
        # Map sorted indices - GMM internal indices
        agb_internal_idxs  = [int(order[i]) for i in agb_sorted_idxs]
        blue_internal_idxs = [int(order[i]) for i in blue_sorted_idxs]
 
        posteriors = gmm.predict_proba(c.reshape(-1, 1))  # shape (N_slice, n_comp)
 
        # RSG: single component
        rsg_post = posteriors[:, rsg_internal_idx]
        mask_rsg = rsg_post >= prob_threshold
        rsg_indices.extend(slice_mask[mask_rsg].tolist())
        rsg_probs.extend(rsg_post[mask_rsg].tolist())
 
        # AGB: sum posteriors across all redder components
        if agb_internal_idxs:
            agb_post = posteriors[:, agb_internal_idxs].sum(axis=1)
            mask_agb = agb_post >= prob_threshold
            luminous_mask = m_ > lcut
            # Exclude stars already claimed by RSG to avoid double-counting
            mask_agb &= ~mask_rsg
            rsg_in_agb = np.copy(mask_agb)
            mask_agb &= luminous_mask
            rsg_in_agb &= ~luminous_mask
            agb_indices.extend(slice_mask[mask_agb].tolist())
            agb_probs.extend(agb_post[mask_agb].tolist())

            rsg_indices.extend(slice_mask[rsg_in_agb].tolist())
            rsg_probs.extend(agb_post[rsg_in_agb].tolist())
 
        # Blue: sum posteriors across all bluer components
        if blue_internal_idxs:
            blue_post = posteriors[:, blue_internal_idxs].sum(axis=1)
            mask_blue = blue_post >= prob_threshold
            mask_blue &= ~mask_rsg
            if agb_internal_idxs:
                mask_blue &= ~mask_agb
            
            #stars 2 sigma bluer than rsgs should be included in the blue sample even if they have low blue_post, since GMM can fail to separate them
            blue_color_cut = rsg_mean - 3 * rsg_sig
            missed_blue_mask = (c < blue_color_cut)
            mask_blue |= missed_blue_mask
            blue_indices.extend(slice_mask[mask_blue].tolist())
            blue_probs.extend(blue_post[mask_blue].tolist())

            # remove these stars from the RSG sample if they were included due to high AGB posterior
            rsg_indices = [idx for idx in rsg_indices if idx not in slice_mask[missed_blue_mask]]
            rsg_probs = [prob for idx, prob in zip(rsg_indices, rsg_probs) if idx not in slice_mask[missed_blue_mask]]
        
 
    return (
        np.array(rsg_indices,  dtype=int), np.array(agb_indices,  dtype=int),
        np.array(blue_indices, dtype=int), np.array(rsg_probs),
        np.array(agb_probs),               np.array(blue_probs),
    )


def plot_rsg_chain_on_cmd(validated, mag, color, rsg_indices=None, ax=None):
    """
    Overplot the chained RSG component means (±1σ) on the CMD.
    Optionally highlight the extracted seed stars.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    ax.scatter(color, mag, s=1, c="grey", alpha=0.3, rasterized=True,
               label="All stars")

    if rsg_indices is not None and len(rsg_indices):
        ax.scatter(color[rsg_indices], mag[rsg_indices],
                   s=4, c="tomato", alpha=0.6, rasterized=True,
                   label="RSG seed")

    for cand in validated.values():
        if cand is None:
            continue
        color_flag = ("orange" if cand.get("recovered")
                      else ("purple" if cand.get("interpolated") else "red"))
        ax.errorbar(cand["mean"], cand["mag_mid"],
                    xerr=cand.get("sigma", 0),
                    fmt="o", color=color_flag,
                    markersize=5, elinewidth=1.5, alpha=0.85)

    ax.invert_yaxis()
    ax.set_xlabel("F115W - F200W")
    ax.set_ylabel("F200W)")
    ax.set_title("Chained RSG seed")
    if rsg_indices is not None:
        ax.legend(markerscale=4, fontsize=8)
    return fig, ax

def plot_chain_on_cmd(validated, mag, color,
                      rsg_indices=None, agb_indices=None,
                      blue_indices=None, ax=None):
    """
    Overplot the chained RSG component means (±1σ) on the CMD.
    Optionally highlight RSG, AGB, and blue seed stars with distinct colours.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure
 
    ax.scatter(color, mag, s=1, c="grey", alpha=0.3, rasterized=True,
               label="All stars")
 
    if blue_indices is not None and len(blue_indices):
        ax.scatter(color[blue_indices], mag[blue_indices],
                   s=4, c="steelblue", alpha=0.6, rasterized=True,
                   label="Blue seed")
 
    if rsg_indices is not None and len(rsg_indices):
        ax.scatter(color[rsg_indices], mag[rsg_indices],
                   s=4, c="tomato", alpha=0.6, rasterized=True,
                   label="RSG seed")
 
    if agb_indices is not None and len(agb_indices):
        ax.scatter(color[agb_indices], mag[agb_indices],
                   s=4, c="firebrick", alpha=0.6, rasterized=True,
                   label="AGB seed")
 
    # RSG chain component markers
    for cand in validated.values():
        if cand is None:
            continue
        chain_color = ("orange" if cand.get("recovered")
                       else ("purple" if cand.get("interpolated") else "red"))
        ax.errorbar(cand["mean"], cand["mag_mid"],
                    xerr=cand.get("sigma", 0),
                    fmt="o", color=chain_color,
                    markersize=5, elinewidth=1.5, alpha=0.85, zorder=5)
 
    ax.invert_yaxis()
    ax.set_xlabel("F115W - F200W")
    ax.set_ylabel("F200W")
    ax.set_title("Seed populations")
    ax.legend(markerscale=4, fontsize=8)
    return fig, ax


def plot_component_tracking(results, validated=None, ax=None):
    """
    Show all GMM component means vs magnitude, with the validated RSG chain
    highlighted in red.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    for key, res in results.items():
        if res is None:
            continue
        m = res["mag_mid"]
        for mean, sigma, w in zip(res["means"], res["sigmas"], res["weights"]):
            ax.scatter(mean, m, s=200 * w, alpha=0.4,
                       c="steelblue", edgecolors="k", linewidths=0.4)

    if validated is not None:
        for cand in validated.values():
            if cand is None:
                continue
            ax.scatter(cand["mean"], cand["mag_mid"],
                       s=80, c="red", zorder=5, marker="*")

    ax.invert_yaxis()
    ax.set_xlabel("F115W  F200W")
    ax.set_ylabel("F200W")
    ax.set_title("GMM components")
    return fig, ax


def plot_bic_curves(results, n_cols=4):
    """
    Grid of BIC-vs-n curves, one panel per magnitude slice.
    """
    valid = [(k, v) for k, v in results.items() if v is not None]
    n_panels = len(valid)
    n_rows   = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3 * n_cols, 2.5 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_visible(False)

    for i, (key, res) in enumerate(valid):
        ax = axes_flat[i]
        ax.set_visible(True)
        ax.plot(res["n_range"], res["bics"], "o-", ms=4)
        best_n = res["n_best"]
        best_bic = res["bics"][res["n_range"].index(best_n)]
        ax.axvline(best_n, color="red", lw=1, ls="--")
        ax.set_title(f"m={res['mag_mid']:.2f}  n={best_n}", fontsize=8)
        ax.set_xlabel("N", fontsize=7)
        ax.set_ylabel("BIC", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    return fig


def run_rsg_pipeline(
    mag,
    color,
    mag_bins,
    lcut,
    exp_rsg_color=0.5,
    n_max=7,
    min_stars=30,
    n_init=7,
    min_weight=0.02,
    trend_deg=2,
    sigma_clip=2.0,
    max_residual=0.12,
    prob_threshold=0.5,
    plot=True,
    save_plots=False,
    plot_prefix="rsg",
):
    """
    End-to-end RSG seed selection pipeline.

    Parameters
    ----------
    mag, color      : (N,) arrays — magnitude and colour of all sources
    mag_bins        : bin edges for magnitude slicing
    n_max           : maximum GMM components per slice
    min_stars       : skip slices with fewer stars
    n_init          : GMM random restarts
    min_weight      : ignore components below this weight
    trend_deg       : polynomial degree for colour-trend fit
    sigma_clip      : sigma-clipping threshold for trend validation
    max_residual    : max |colour - trend| allowed in recovery step
    prob_threshold  : minimum RSG posterior probability to keep a star
    plot            : if True, produce diagnostic figures
    save_plots      : if True, save figures to disk
    plot_prefix     : filename prefix when save_plots=True

    Returns
    -------
    rsg_indices : (M,) int array   — indices of seed RSG stars
    rsg_probs   : (M,) float array — RSG membership probability per star
    validated   : per-slice chain dict (for inspection / debugging)
    trend_fn    : callable mag → predicted RSG colour
    results     : raw per-slice GMM results
    """
    print("Step 1/4  Fitting GMMs per magnitude slice …")
    results = slice_and_fit(mag, color, mag_bins,
                             n_max=n_max, min_stars=min_stars, n_init=n_init)
    n_fitted = sum(1 for v in results.values() if v is not None)
    print(f"          {n_fitted} slices fitted (of {len(results)} total)")

    print("Step 2/4  Identifying RSG candidate per slice (gap heuristic) …")
    candidates = initialize_rsg_chain(results, min_weight=min_weight, exp_rsg_color=exp_rsg_color)

    print("Step 3/4  Fitting colour trend and sigma-clipping outlier slices …")
    validated, trend_fn = fit_and_validate_chain(
        candidates, deg=trend_deg, sigma_clip=sigma_clip
    )
    n_on    = sum(1 for v in validated.values()
                  if v is not None and v.get("on_trend"))
    n_off   = sum(1 for v in validated.values()
                  if v is not None and not v.get("on_trend"))
    print(f"          {n_on} on-trend  |  {n_off} off-trend (attempting recovery)")

    validated = recover_merged_slices(
        validated, results, trend_fn,
        max_residual=max_residual, min_weight=min_weight
    )
    n_rec   = sum(1 for v in validated.values() if v and v.get("recovered"))
    n_interp = sum(1 for v in validated.values() if v and v.get("interpolated"))
    print(f"          {n_rec} recovered  |  {n_interp} interpolated from trend")

    print("Step 4/4  Extracting RSG seed stars via GMM posteriors …")
    rsg_indices, agb_indices, blue_indices, rsg_probs, agb_probs, blue_probs = \
        extract_seeds(validated, results, mag, color, lcut, prob_threshold=prob_threshold)
    print(f"          RSGs : {len(rsg_indices)}  |  "
          f"AGBs : {len(agb_indices)}  |  "
          f"Blue : {len(blue_indices)}  "
          f"(p ≥ {prob_threshold})")

    if plot:
        fig1, _ = plot_chain_on_cmd(validated, mag, color,
                            rsg_indices, agb_indices, blue_indices)
        fig2, _ = plot_component_tracking(results, validated)
        fig3    = plot_bic_curves(results)

        if save_plots:
            fig1.savefig(f"{plot_prefix}_cmd_chain.png",   dpi=150, bbox_inches="tight")
            fig2.savefig(f"{plot_prefix}_component_track.png", dpi=150, bbox_inches="tight")
            fig3.savefig(f"{plot_prefix}_bic_curves.png",  dpi=150, bbox_inches="tight")
            print(f"Plots saved with prefix '{plot_prefix}_'")

        plt.show()

    return (rsg_indices, agb_indices, blue_indices,
            rsg_probs, agb_probs, blue_probs,
            validated, trend_fn, results)