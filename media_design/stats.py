import numpy as np
import pandas as pd


def percent_diff_per_ion(
    all_ions: list[str],
    ion_contributions: np.ndarray,
    geochemical_data: dict,
    *,
    display: bool = True,
    eps: float = 1e-30,                 # <-- BACKCOMPAT: used by calculate_similarity
    near_zero_pct_tol: float = 1e-4,    # 0.0001% treated as "≈0%"
    missing_actual_tol: float = 1e-15,  # actual <= this => "missing from solution"
) -> pd.DataFrame:
    """
    Compute per-ion differences between target and modeled ion concentrations.

    Parameters
    ----------
    all_ions : list of str
        Ordered list of all ions represented in `ion_contributions`. This list
        is used to map ion names to indices in the contribution array.
    ion_contributions : numpy.ndarray
        Array of modeled ion concentrations corresponding to `all_ions`.
    geochemical_data : dict
        Dictionary mapping target ion names to target concentrations.
    display : bool, optional
        If True, print a readable summary of per-ion differences. Default is
        True.
    eps : float, optional
        Small value added to the target concentration denominator when
        calculating percent difference to avoid divide-by-zero for extremely
        small targets. Default is 1e-30.
    near_zero_pct_tol : float, optional
        Absolute percent-difference threshold below which an ion is grouped and
        printed as approximately zero difference. Default is 1e-4 (%).
    missing_actual_tol : float, optional
        Absolute actual-concentration threshold below which an ion is treated as
        effectively missing from the solution for display purposes. Default is
        1e-15.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per target ion with the following columns:

        - 'ion'
            Ion name
        - 'target'
            Target concentration
        - 'actual'
            Modeled concentration
        - 'abs_diff'
            Signed difference (`actual - target`)
        - 'abs_abs_diff'
            Absolute difference (`|actual - target|`)
        - 'percent_diff'
            Signed percent difference relative to the target

    Notes
    -----
    - Percent difference is calculated as:

      `100 * (actual - target) / (abs(target) + eps)`

    - `eps` is retained for compatibility with downstream functions such as
      `calculate_similarity`.
    - Ions are only excluded from the output if their target concentration is
      less than or equal to zero.
    - "Missing from solution" is a display-only classification meaning:
      `target > 0` and `abs(actual) <= missing_actual_tol`.
    
    """

    ion_index = {ion: i for i, ion in enumerate(all_ions)}

    rows = []
    for ion, target_val in geochemical_data.items():
        target = float(target_val)
        if target <= 0:
            continue

        actual = 0.0
        if ion in ion_index:
            actual = float(ion_contributions[ion_index[ion]])

        abs_diff = actual - target
        abs_abs_diff = abs(abs_diff)

        denom = abs(target) + float(eps)
        pct_diff = 100.0 * (abs_diff / denom)

        rows.append(
            {
                "ion": ion,
                "target": target,
                "actual": actual,
                "abs_diff": abs_diff,
                "abs_abs_diff": abs_abs_diff,
                "percent_diff": pct_diff,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        if display:
            print("\n== Percent Difference per Ion (Readable) ==\n(no target ions found)")
        return df

    # Classification for printing only (does not affect returned df)
    missing_mask = (df["actual"].abs() <= float(missing_actual_tol))
    near_zero_mask = (~missing_mask) & (df["percent_diff"].abs() < float(near_zero_pct_tol))

    missing_ions = df.loc[missing_mask, "ion"].tolist()
    near_zero_ions = df.loc[near_zero_mask, "ion"].tolist()

    other_df = df.loc[~missing_mask & ~near_zero_mask].copy()
    other_df = other_df.sort_values("percent_diff", key=lambda s: s.abs(), ascending=False)

    if display:
        print("\n== Percent Difference per Ion (Readable) ==")

        if near_zero_ions:
            print(f"\n≈0% diff (|%diff| < {near_zero_pct_tol:g}%):")
            print(", ".join(near_zero_ions))

        if missing_ions:
            print("\nMissing from solution (target > 0, actual ~ 0):")
            print(", ".join(missing_ions))

        if not other_df.empty:
            print("\nOther ions:")
            for _, r in other_df.iterrows():
                print(f"{r['ion']}: {r['percent_diff']:+.3f}%")

    return df


def calculate_similarity(
    all_ions: list[str],
    ion_contributions: np.ndarray,
    geochemical_data: dict,
    *,
    display: bool = True,
    eps: float = 1e-30,
    print_decimals: int = 6,   # controls display only
) -> float:
    """
    Calculate overall similarity between modeled and target ion compositions.

    Parameters
    ----------
    all_ions : list of str
        Ordered list of all ions represented in `ion_contributions`. This list
        is used to map ion names to indices in the contribution array.
    ion_contributions : numpy.ndarray
        Array of modeled ion concentrations corresponding to `all_ions`.
    geochemical_data : dict
        Dictionary mapping target ion names to target concentrations.
    display : bool, optional
        If True, print the overall similarity as a percentage. Default is True.
    eps : float, optional
        Small value added to the denominator to avoid divide-by-zero when the
        total target magnitude is extremely small. Default is 1e-30.
    print_decimals : int, optional
        Number of decimal places to use when printing the similarity
        percentage. This affects display only and does not alter the returned
        value. Default is 6.

    Returns
    -------
    float
        Raw overall similarity value computed as:

        `1 - (sum |actual - target|) / (sum |target|)`

    Notes
    -----
    - This is a weighted overall similarity because each ion contributes to the
      numerator and denominator according to its absolute target concentration.
    - The returned value is not rounded and is not constrained to a fixed
      range; poor fits may produce values below zero.
    - `print_decimals` affects only the printed output, not the returned float.
    - Because the calculation reuses `percent_diff_per_ion`, only ions with
      positive target concentrations are included.
   
    """

    # reuse your percent-diff dataframe (now eps-safe)
    df = percent_diff_per_ion(
        all_ions,
        ion_contributions,
        geochemical_data,
        display=False,
        eps=eps,
    )

    denom = float(df["target"].abs().sum()) + eps
    numer = float(df["abs_abs_diff"].sum())

    similarity = 1.0 - (numer / denom)

    # IMPORTANT: do NOT clamp or round
    similarity = float(similarity)

    if display:
        pct = similarity * 100.0
        fmt = f"{{:.{print_decimals}f}}"
        print(f"\nOverall similarity to original target: {fmt.format(pct)}%")

    return similarity