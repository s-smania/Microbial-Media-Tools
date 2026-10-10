import numpy as np
import pandas as pd
from scipy.optimize import linprog

# internal defaults (edit here if you want to increase or decrease the penalties)
_DEFAULTS = {
    "penalty_over_trace": 1e12,
    "penalty_over_nontrace": 1e6,
    "penalty_under_trace": 1e12,
    "penalty_under_nontrace": 1e4,
    "hard_min_frac": 0.1,
}

def solve_media_lp(
    reagents: dict,
    geochemical_data,
    *,
    trace_elements: set | None = None,
    hard_min_ions: set | None = None,
    settings: dict | None = None,
):
    """
    Solve the media formulation problem as a linear program with under- and over-shoot slack.

    Parameters
    ----------
    reagents : dict
        Reagent definition dictionary in the standard format:

        .. code-block:: python

            {
                "NaCl": {
                    "ions": {"Na+": 1, "Cl-": 1},
                    "molar_mass": 58.44
                },
                ...
            }

        Each reagent must include:
        - 'ions' : dict mapping ion names to stoichiometric coefficients
        - 'molar_mass' : reagent molar mass in g/mol
    geochemical_data : dict or str
        Target ion concentrations. Supported inputs are:

        - dict mapping ion names to target concentrations
        - path to a CSV file containing either:
          - 'ion' and 'concentration' columns, or
          - columns whose names contain '(Molality)', with concentrations in
            the first row
    trace_elements : set or None, optional
        Set of ions to treat as trace elements. Trace ions receive stronger
        under- and over-shoot penalties according to the active settings.
        Default is None.
    hard_min_ions : set or None, optional
        Set of ions for which a minimum fraction of the target concentration
        must be achieved. The required minimum fraction is controlled by
        `settings["hard_min_frac"]`. Default is None.
    settings : dict or None, optional
        Optional dictionary of solver settings that override internal defaults.
        Supported keys include:

        - 'penalty_over_trace'
        - 'penalty_over_nontrace'
        - 'penalty_under_trace'
        - 'penalty_under_nontrace'
        - 'hard_min_frac'
        - 'penalty_offtarget'

        Any omitted keys fall back to `_DEFAULTS`.

    Returns
    -------
    tuple
        Tuple containing:

        - res : scipy.optimize.OptimizeResult
            Result object returned by `scipy.optimize.linprog`
        - all_ions : list of str
            All ions represented across the reagent set
        - target_ions : list of str
            Target ions present in `geochemical_data`
        - A : numpy.ndarray
            Full stoichiometric matrix with rows as ions and columns as reagents
        - target_idx : list of int
            Indices of target ions within `all_ions`
        - offtarget_idx : list of int
            Indices of non-target ions within `all_ions`
        - reagent_names : list of str
            Ordered list of reagent names corresponding to matrix columns

    Notes
    -----
    - Reagent variables are constrained to be nonnegative.
    - Undershoot and overshoot are handled with separate slack variables,
      allowing asymmetric penalties.
    - Off-target ions are penalized indirectly through reagent costs when a
      reagent contributes ions not present in the target solution.
    - Trace elements can be penalized much more strongly than non-trace ions.
    - Hard minimum constraints require certain ions to reach at least a fixed
      fraction of their target value.
    """

    if isinstance(geochemical_data, str):
        df = pd.read_csv(geochemical_data)
    
        # Case 1: ion / concentration format
        if {"ion", "concentration"}.issubset(df.columns):
            geochemical_data = dict(
                zip(df["ion"].astype(str), pd.to_numeric(df["concentration"], errors="coerce"))
            )
    
        # Case 2: formatted_df style (Molality columns)
        else:
            ion_cols = [c for c in df.columns if "(Molality)" in c]
    
            geochemical_data = {
                c.split()[0]: float(df.iloc[0][c])
                for c in ion_cols
                if pd.notnull(df.iloc[0][c])
        }

    if trace_elements is None:
        trace_elements = set()
    if hard_min_ions is None:
        hard_min_ions = set()

    cfg = dict(_DEFAULTS)
    if settings:
        cfg.update(settings)

    # Build list of all ions from reagents
    all_ions: list[str] = []
    for r_data in reagents.values():
        for ion in r_data["ions"].keys():
            if ion not in all_ions:
                all_ions.append(ion)

    target_ions = list(geochemical_data.keys())

    missing_ions = [ion for ion in target_ions if ion not in all_ions]
    if missing_ions:
        raise ValueError(f"Missing ion coverage for: {', '.join(missing_ions)}")

    reagent_names = list(reagents.keys())
    A = np.array(
        [[reagents[r]["ions"].get(ion, 0.0) for r in reagent_names] for ion in all_ions],
        dtype=float,
    )

    target_idx = [all_ions.index(i) for i in target_ions]
    offtarget_idx = [i for i in range(len(all_ions)) if i not in target_idx]

    A_target = A[target_idx]
    b_target = np.array([geochemical_data[i] for i in target_ions], dtype=float)

    n_reagents = A.shape[1]
    n_targets = len(target_ions)

    # Variables: [x, s_under, s_over]
    # Undershoot: A x + s_under >= b -> -(A x + s_under) <= -b
    A_under = np.hstack([-A_target, -np.eye(n_targets), np.zeros((n_targets, n_targets))])
    b_under = -b_target

    # Overshoot: A x - s_over <= b
    A_over = np.hstack([A_target, np.zeros((n_targets, n_targets)), -np.eye(n_targets)])
    b_over = b_target

    A_ub = np.vstack([A_under, A_over])
    b_ub = np.concatenate([b_under, b_over])

    # Hard minimums: A_hard x >= frac*b_hard -> -A_hard x <= -frac*b_hard
    if hard_min_ions:
        hard_list = [ion for ion in target_ions if ion in hard_min_ions]
        if hard_list:
            hard_rows = [target_ions.index(ion) for ion in hard_list]
            A_hard = A_target[hard_rows]
            b_hard = b_target[hard_rows] * float(cfg["hard_min_frac"])

            A_hard_full = np.hstack(
                [
                    -A_hard,
                    np.zeros((len(hard_rows), n_targets)),
                    np.zeros((len(hard_rows), n_targets)),
                ]
            )
            b_hard_full = -b_hard
            A_ub = np.vstack([A_ub, A_hard_full])
            b_ub = np.concatenate([b_ub, b_hard_full])

    # Base reagent cost (minimize total mass)
    base_cost = np.array(
        [reagents[r]["molar_mass"] for r in reagent_names],
        dtype=float
    )
    # Penalize reagents that generate ions not present in the target solution.
    off_penalty = float(cfg.get("penalty_offtarget", 1e8))  # default strong penalty

    if offtarget_idx:
        # mmol of off-target ions produced per mmol reagent
        off_per_reagent = A[offtarget_idx, :].sum(axis=0)
    else:
        off_per_reagent = np.zeros(n_reagents, dtype=float)

    # Add penalty directly into objective coefficient
    base_cost = base_cost + off_penalty * off_per_reagent

    
    under_penalty = np.array(
        [
            cfg["penalty_under_trace"] if ion in trace_elements else cfg["penalty_under_nontrace"]
            for ion in target_ions
        ],
        dtype=float,
    )
    over_penalty = np.array(
        [
            cfg["penalty_over_trace"] if ion in trace_elements else cfg["penalty_over_nontrace"]
            for ion in target_ions
        ],
        dtype=float,
    )

    c = np.concatenate([base_cost, under_penalty, over_penalty])
    bounds = [(0, None)] * (n_reagents + 2 * n_targets)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    return res, all_ions, target_ions, A, target_idx, offtarget_idx, reagent_names


def split_geochem_targets(
    geochem_dict: dict[str, float],
    *,
    trace_threshold: float = 1e-6,
    trace_ions: list[str] | None = None,
):
    """
    Split geochemical target ions into base and trace groups.

    Parameters
    ----------
    geochem_dict : dict of {str : float}
        Dictionary mapping ion names to target concentrations.
    trace_threshold : float, optional
        Concentration threshold used to classify ions as trace when
        `trace_ions` is not provided. Ions with concentrations less than or
        equal to this threshold are placed in the trace dictionary. Default is
        1e-6.
    trace_ions : list of str or None, optional
        Explicit list of ions to classify as trace. If provided and non-empty,
        this list overrides threshold-based classification. Default is None.

    Returns
    -------
    tuple
        Tuple containing:

        - base : dict
            Dictionary of non-trace ion targets
        - trace : dict
            Dictionary of trace ion targets

    Notes
    -----
    - Explicit `trace_ions` takes precedence over threshold-based splitting.
    - Output values are converted to float.
    - Ions equal to the threshold are classified as trace.
    """
    
    if trace_ions is not None and len(trace_ions) > 0:
        trace_set = set(trace_ions)
        base = {ion: float(v) for ion, v in geochem_dict.items() if ion not in trace_set}
        trace = {ion: float(v) for ion, v in geochem_dict.items() if ion in trace_set}
        return base, trace

    base, trace = {}, {}
    thr = float(trace_threshold)
    for ion, v in geochem_dict.items():
        v = float(v)
        (trace if v <= thr else base)[ion] = v
    return base, trace


def solve_two_recipe_media(
    reagents: dict,
    geochem_dict: dict[str, float],
    *,
    trace_threshold: float = 1e-6,
    trace_ions: list[str] | None = None,  # [] => threshold mode
    enforce_trace_hardmin: bool = True,
):
    """
    Solve separate linear programs for base-media and trace-media targets.

    Parameters
    ----------
    reagents : dict
        Reagent definition dictionary in the standard nested format expected by
        `solve_media_lp`.
    geochem_dict : dict of {str : float}
        Full target ion composition dictionary.
    trace_threshold : float, optional
        Concentration threshold used to separate trace ions from base ions when
        `trace_ions` is not provided. Default is 1e-6.
    trace_ions : list of str or None, optional
        Explicit list of ions to classify as trace. If provided and non-empty,
        this overrides threshold-based classification. Default is None.
    enforce_trace_hardmin : bool, optional
        If True, require trace ions to meet the hard-minimum fraction in the
        trace solve. Default is True.

    Returns
    -------
    dict
        Dictionary containing:

        - 'geochem_base' : dict
            Base ion targets
        - 'geochem_trace' : dict
            Trace ion targets
        - 'base' : tuple
            Output tuple returned by `solve_media_lp` for the base solve
        - 'trace' : tuple
            Output tuple returned by `solve_media_lp` for the trace solve

    Notes
    -----
    - The base solve treats ions as standard targets with default penalties.
    - The trace solve marks all trace ions as `trace_elements` so that trace
      penalties are applied.
    - This function does not scale trace targets into a stock solution; it
      simply solves two independent target sets.
    """
    
    geochem_base, geochem_trace = split_geochem_targets(
        geochem_dict,
        trace_threshold=trace_threshold,
        trace_ions=trace_ions,
    )

    base_pack = solve_media_lp(reagents, geochem_base)

    hardmins = set(geochem_trace.keys()) if enforce_trace_hardmin else set()
    trace_pack = solve_media_lp(
        reagents,
        geochem_trace,
        trace_elements=set(geochem_trace.keys()),
        hard_min_ions=hardmins,
    )

    return {
        "geochem_base": geochem_base,
        "geochem_trace": geochem_trace,
        "base": base_pack,
        "trace": trace_pack,
    }


def create_recipe(
    reagents: dict,
    geochem_dict: dict,
    *,
    trace_threshold: float = 1e-6,
    trace_ions: list[str] | None = None,     # []/None => threshold mode
    trace_stock_mL_per_L: float = 1.0,       # 1 mL into 1 L = 1000×
    enforce_trace_hardmin: bool = True,
    reagent_cutoff_g_L: float = 0.0,
):
    """
    Generate a two-part media recipe consisting of a base medium and a trace stock.

    Parameters
    ----------
    reagents : dict
        Reagent definition dictionary in the standard nested format expected by
        `solve_media_lp`.
    geochem_dict : dict
        Full target ion composition dictionary mapping ion names to final target
        concentrations.
    trace_threshold : float, optional
        Concentration threshold used to classify ions as trace when
        `trace_ions` is not provided. Default is 1e-6.
    trace_ions : list of str or None, optional
        Explicit list of ions to classify as trace. If provided and non-empty,
        this overrides threshold-based classification. Default is None.
    trace_stock_mL_per_L : float, optional
        Trace stock addition volume in mL per L of final medium. Used to convert
        final trace targets into stock concentrations. For example, 1 mL/L
        corresponds to a 1000× stock. Default is 1.0.
    enforce_trace_hardmin : bool, optional
        If True, apply hard-minimum constraints to trace ions in the trace stock
        solve. Default is True.
    reagent_cutoff_g_L : float, optional
        Minimum reagent concentration in g/L required for inclusion in the
        output reagent tables. Reagents at or below this threshold are omitted
        from the returned recipe DataFrames. Default is 0.0.

    Returns
    -------
    tuple
        Tuple containing:

        - base_df : pandas.DataFrame
            Base-medium reagent recipe with columns:
            - 'reagent'
            - 'mmol_per_L_final'
            - 'g_per_L_final'

        - trace_stock_df : pandas.DataFrame
            Trace-stock reagent recipe with columns:
            - 'reagent'
            - 'mmol_per_L_stock'
            - 'g_per_L_stock'

        - ions_total_df : pandas.DataFrame
            Combined ion concentrations in the final medium with columns:
            - 'ion'
            - 'base_mmol_L_final'
            - 'trace_mmol_L_final'
            - 'total_mmol_L_final'
            - 'target_mmol_L_final'

        - meta : dict
            Dictionary of metadata and raw solver outputs, including:
            - stock and dilution factors
            - base and trace target dictionaries
            - solver result objects
            - ion contribution arrays

    Notes
    -----
    - A stock made at `trace_stock_mL_per_L = 1.0` mL/L is treated as a 1000×
      stock.
    - Base ions are solved directly at final media concentrations.
    - Trace ions are solved at stock concentrations, then scaled back into final
      media contributions.
    - Reagent masses are reported in g/L using reagent molar masses.
    - Reagent rows can be filtered from the output tables using
      `reagent_cutoff_g_L`.
    - The function assumes that base and trace solves produce the same
      `all_ions` ordering because both are derived from the same reagent set.
    """
    
    if trace_stock_mL_per_L <= 0:
        raise ValueError("trace_stock_mL_per_L must be > 0")

    # dilution factor: stock is diluted by (mL/L)/1000
    # so stock concentration = final / (mL/L / 1000) = final * (1000 / (mL/L))
    stock_factor = 1000.0 / float(trace_stock_mL_per_L)
    dilution = 1.0 / stock_factor  # fraction of stock in final media

    # Split ions into base vs trace
    geochem_base, geochem_trace_final = split_geochem_targets(
        geochem_dict,
        trace_threshold=trace_threshold,
        trace_ions=trace_ions,
    )

    # Scale trace targets to STOCK concentrations
    geochem_trace_stock = {ion: float(v) * stock_factor for ion, v in geochem_trace_final.items()}

    # Solve BASE
    res_b, all_ions_b, target_ions_b, A_b, target_idx_b, off_b, reagent_names_b = solve_media_lp(
        reagents,
        geochem_base,
    )
    if not res_b.success:
        raise RuntimeError(f"Base LP failed: {res_b.message}")

    # Solve TRACE (as STOCK)
    hardmins = set(geochem_trace_stock.keys()) if enforce_trace_hardmin else set()
    res_t, all_ions_t, target_ions_t, A_t, target_idx_t, off_t, reagent_names_t = solve_media_lp(
        reagents,
        geochem_trace_stock,
        trace_elements=set(geochem_trace_stock.keys()),
        hard_min_ions=hardmins,
    )
    if not res_t.success:
        raise RuntimeError(f"Trace-stock LP failed: {res_t.message}")

    # Ensure consistent ion ordering (should match because derived from reagents)
    if all_ions_b != all_ions_t:
        raise ValueError("all_ions mismatch between base and trace solves (unexpected).")

    def _reagent_df(res, reagent_names, *, mmol_col: str, g_col: str):
        x = res.x[:len(reagent_names)]
        df = pd.DataFrame({
            "reagent": reagent_names,
            mmol_col: x,
            g_col: [float(xi) * float(reagents[name]["molar_mass"]) / 1000.0
                    for xi, name in zip(x, reagent_names)],
        })
        if reagent_cutoff_g_L > 0:
            df = df[df[g_col] > reagent_cutoff_g_L].reset_index(drop=True)
        return df

    def _ion_vector(res, A):
        x = res.x[:A.shape[1]]
        return A @ x

    # Reagent dataframes
    base_df = _reagent_df(res_b, reagent_names_b, mmol_col="mmol_per_L_final", g_col="g_per_L_final")
    trace_stock_df = _reagent_df(res_t, reagent_names_t, mmol_col="mmol_per_L_stock", g_col="g_per_L_stock")

    # Ion contributions
    ions_base_final = _ion_vector(res_b, A_b)                 # mmol/L final
    ions_trace_stock = _ion_vector(res_t, A_t)                # mmol/L in stock
    ions_trace_final = ions_trace_stock * dilution            # mmol/L final contributed by dosing
    ions_total_final = ions_base_final + ions_trace_final

    ions_total_df = pd.DataFrame({
        "ion": all_ions_b,
        "base_mmol_L_final": ions_base_final,
        "trace_mmol_L_final": ions_trace_final,
        "total_mmol_L_final": ions_total_final,
        "target_mmol_L_final": [float(geochem_dict.get(ion, 0.0)) for ion in all_ions_b],
    })

    meta = {
        "stock_factor": stock_factor,
        "dilution_fraction": dilution,
        "geochem_base": geochem_base,
        "geochem_trace_final": geochem_trace_final,
        "geochem_trace_stock": geochem_trace_stock,
        "base_result": res_b,
        "trace_result": res_t,
        "all_ions": all_ions_b,
        "ions_base_final": ions_base_final,
        "ions_trace_stock": ions_trace_stock,
        "ions_trace_final": ions_trace_final,
        "ions_total_final": ions_total_final,
    }

    return base_df, trace_stock_df, ions_total_df, meta