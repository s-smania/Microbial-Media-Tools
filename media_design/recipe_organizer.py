import math
import pandas as pd
import numpy as np
from collections import defaultdict

from .paths import resolve_data_path


# Configuration

ROMAN = ["I","II","III","IV","V","VI","VII","VIII","IX","X",
         "XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX"]

DEFAULT_METAL_IONS = {
    "Fe+2","Fe+3","Cu+2","Zn+2","Sn+2","Pb+2","Co+2","Al+3",
    "Cd+2","Ga+3","Hf+4","Ni+2","Mn+2"
}


# Rule parsin

def _parse_rule(rule_val):
    """
    Parse a reagent compatibility/autoclave rule entry.

    Parameters
    ----------
    rule_val : Any
        Rule value read from the reagent rules CSV. Supported forms include:
        - None or NaN: no rule
        - empty string: no rule
        - "No": reagent is non-autoclavable
        - comma-, semicolon-, or pipe-delimited ion list: ions incompatible
          with the reagent in the same subgroup

    Returns
    -------
    tuple
        Tuple containing:

        - bool
            True if the reagent is marked as non-autoclavable, otherwise False
        - set of str
            Set of excluded ions parsed from the rule entry

    """
    if rule_val is None:
        return False, set()
    if isinstance(rule_val, float) and math.isnan(rule_val):
        return False, set()

    s = str(rule_val).strip()
    if s == "":
        return False, set()

    if s.lower() == "no":
        return True, set()

    parts = []
    for chunk in s.replace(";", ",").replace("|", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    return False, set(parts)


# Formatting

def _fmt_mass(x):
    """
    Format a reagent mass value for user-facing recipe output.

    Parameters
    ----------
    x : float
        Mass value to format.

    Returns
    -------
    str
        Mass formatted to three decimal places without units.

    """
    return f"{x:.3f}"

def _fmt_mL(x):
    """
    Format a liquid addition volume for user-facing recipe output.

    Parameters
    ----------
    x : float
        Volume value in mL.

    Returns
    -------
    str
        Volume formatted using general floating-point notation followed by
        ' mL'.

    """
    return f"{x:.6g} mL"


# Master file loading

def _build_master_maps(path):
    """
    Build reagent lookup maps from a master reagent/rules CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the master reagent CSV. A bare file name that is not found in
        the current working directory but that matches a file bundled in the
        package 'data' directory resolves to the bundled copy. The file is
        expected to contain at least the following columns:
        - 'reagent_name'
        - 'ion'
        - 'rule'

    Returns
    -------
    tuple
        Tuple containing:

        - ions_by_reagent : collections.defaultdict
            Mapping of reagent name to a set of ions contributed by that reagent
        - non_autocl : dict
            Mapping of reagent name to a boolean indicating whether the reagent
            is non-autoclavable
        - excluded : dict
            Mapping of reagent name to a set of excluded ions derived from the
            rule field

    """
    df = pd.read_csv(resolve_data_path(path, default="master-reagents.csv"))

    ions_by_reagent = defaultdict(set)
    non_autocl = {}
    excluded = {}

    for _, r in df.iterrows():
        reagent = str(r["reagent_name"]).strip()
        ion = str(r["ion"]).strip()
        if reagent and ion and ion.lower() != "nan":
            ions_by_reagent[reagent].add(ion)

    for _, r in df.iterrows():
        reagent = str(r["reagent_name"]).strip()
        na, ex = _parse_rule(r["rule"])
        non_autocl[reagent] = na
        excluded[reagent] = ex

    return ions_by_reagent, non_autocl, excluded


# Classification
    
def _classify_type(reagent, ions, metal_ions):
    """
    Classify a reagent into a preparation type group.

    Parameters
    ----------
    reagent : str
        Reagent name.
    ions : set
        Set of ions contributed by the reagent.
    metal_ions : set
        Set of ions that should be treated as metals.

    Returns
    -------
    str
        Reagent type classification. Possible values are:
        - 'metals'
        - 'hydrated salts'
        - 'anhydrous salts'

    """
    
    if any(i in metal_ions for i in ions):
        return "metals"
    if "H2O" in reagent:
        return "hydrated salts"
    return "anhydrous salts"

def _can_place(ions, excluded, sg_ions, sg_excluded):
    """
    Determine whether a reagent can be placed into an existing subgroup.

    Parameters
    ----------
    ions : set
        Ion set for the candidate reagent.
    excluded : set
        Excluded ions for the candidate reagent.
    sg_ions : set
        Combined ion set already present in the subgroup.
    sg_excluded : set
        Combined exclusion set already present in the subgroup.

    Returns
    -------
    bool
        True if the candidate reagent is compatible with the subgroup,
        otherwise False.

    """
    if sg_ions & excluded:
        return False
    if ions & sg_excluded:
        return False
    return True


# Stock factor logic


def _find_stock_factor(
    subgroup,
    final_g,
    *,
    stock_volume_L=0.1,
    preferred_min=0.1,
    fallback_min=0.01,
    min_add_mL=0.01,
    max_mass=10.0
):
    """
    Determine a workable stock concentration factor for a subgroup of reagents.

    Parameters
    ----------
    subgroup : list of dict
        List of reagent item dictionaries. Each item must include a 'reagent'
        key corresponding to entries in `final_g`.
    final_g : dict
        Mapping of reagent name to final concentration in g/L.
    stock_volume_L : float, optional
        Volume of stock to prepare, in liters. Default is 0.1 L (100 mL).
    preferred_min : float, optional
        Preferred minimum weighable mass per reagent in the stock preparation,
        in grams. Default is 0.1 g.
    fallback_min : float, optional
        Fallback minimum weighable mass per reagent in the stock preparation,
        in grams. Default is 0.01 g.
    min_add_mL : float, optional
        Minimum acceptable stock addition volume per liter of final media, in
        mL. Default is 0.01 mL.
    max_mass : float, optional
        Maximum allowable mass of any reagent in the stock preparation, in
        grams. Default is 10.0 g.

    Returns
    -------
    tuple
        Tuple containing:

        - int or None
            Stock concentration factor, expressed as an integer multiple of the
            final concentration
        - float or None
            Minimum mass threshold actually used (`preferred_min` or
            `fallback_min`)

    """
    gvals = [final_g[it["reagent"]] for it in subgroup]
    gmin = min(gvals)

    max_factor = 1000.0 / min_add_mL

    def attempt(min_mass):
        required = min_mass / (gmin * stock_volume_L)
        factor = 10 ** math.ceil(math.log10(required))

        if factor > max_factor:
            return None

        for g in gvals:
            if g * factor * stock_volume_L > max_mass:
                return None

        return int(factor)

    factor = attempt(preferred_min)
    if factor:
        return factor, preferred_min

    factor = attempt(fallback_min)
    if factor:
        return factor, fallback_min

    return None, None

def _split_trace_by_mass_window(group_items, preferred_min_mass, max_trace_mass):
    """
    Split a compatible trace group into serial-dilution subgroups with workable masses.

    Parameters
    ----------
    group_items : list of dict
        List of reagent item dictionaries. Each item must include a
        'g_per_L_final' value.
    preferred_min_mass : float
        Minimum desired reagent mass in the 1 L primary stock, in grams.
    max_trace_mass : float
        Maximum allowable reagent mass in the 1 L primary stock, in grams.

    Returns
    -------
    list of tuple
        List of `(sub_items, primary_factor)` tuples where:

        - sub_items : list of dict
            Reagents assigned to the subgroup
        - primary_factor : int
            Power-of-10 primary stock factor used for that subgroup

    """
    items_sorted = sorted(group_items, key=lambda x: x["g_per_L_final"])
    out = []

    cur = []
    for it in items_sorted:
        trial = cur + [it]
        gmin = min(x["g_per_L_final"] for x in trial)
        gmax = max(x["g_per_L_final"] for x in trial)

        # bounds on primary_factor for 1 L primary stock
        lower = preferred_min_mass / gmin
        upper = max_trace_mass / gmax

        # enforce >= 1000 and power-of-10 factor
        lower = max(lower, 1e3)

        if upper < lower:
            # finalize current group
            if not cur:
                # single item can't satisfy constraint (rare but possible)
                # force it into its own group with primary_factor=1000
                out.append(([it], 1000))
                cur = []
                continue

            gmin_c = min(x["g_per_L_final"] for x in cur)
            gmax_c = max(x["g_per_L_final"] for x in cur)
            lower_c = max(preferred_min_mass / gmin_c, 1e3)
            upper_c = max_trace_mass / gmax_c

            n_low = math.ceil(math.log10(lower_c))
            n_high = math.floor(math.log10(upper_c))
            # choose the largest power-of-10 that fits (maximizes dilution)
            n = n_high if n_high >= n_low else n_low
            primary_factor = int(10 ** n)

            out.append((cur, primary_factor))
            cur = [it]
        else:
            cur = trial

    if cur:
        gmin_c = min(x["g_per_L_final"] for x in cur)
        gmax_c = max(x["g_per_L_final"] for x in cur)
        lower_c = max(preferred_min_mass / gmin_c, 1e3)
        upper_c = max_trace_mass / gmax_c

        n_low = math.ceil(math.log10(lower_c))
        n_high = math.floor(math.log10(upper_c))
        n = n_high if n_high >= n_low else n_low
        primary_factor = int(10 ** n)

        out.append((cur, primary_factor))

    return out
    

# Trace merge

def merge_trace_into_base(base_df, trace_stock_df, trace_stock_mL_per_L=1.0):
    """
    Merge a trace stock recipe into a base recipe at final media concentrations.

    Parameters
    ----------
    base_df : pandas.DataFrame
        Base recipe DataFrame containing at least:
        - 'reagent'
        - 'g_per_L_final'
    trace_stock_df : pandas.DataFrame
        Trace stock recipe DataFrame containing at least:
        - 'reagent'
        - 'g_per_L_stock'
    trace_stock_mL_per_L : float, optional
        Trace stock dosing volume in mL per liter of final media. Default is
        1.0 mL/L.

    Returns
    -------
    pandas.DataFrame
        Merged recipe DataFrame with columns:
        - 'reagent'
        - 'g_per_L_final'

    """
    trace_final = trace_stock_df.copy()

    trace_final["g_per_L_final"] = (
        trace_final["g_per_L_stock"] * trace_stock_mL_per_L / 1000.0
    )

    trace_final = trace_final[["reagent", "g_per_L_final"]]

    merged = pd.concat([base_df, trace_final], ignore_index=True)


    # Collapse duplicate reagents by summing g_per_L_final

    merged["g_per_L_final"] = pd.to_numeric(
        merged["g_per_L_final"], errors="coerce"
    ).fillna(0.0)

    merged = (
        merged
        .groupby("reagent", as_index=False)
        .agg({"g_per_L_final": "sum"})
        .sort_values("reagent")
        .reset_index(drop=True)
    )

    return merged


def _group_sort_value(group):
    """Return largest final g/L value in a grouped item collection."""
    return max((it["g_per_L_final"] for it in group.get("items", [])), default=0.0)

def _items_desc(items):
    """Sort reagent items from largest to smallest final g/L value."""
    return sorted(items, key=lambda x: x["g_per_L_final"], reverse=True)


# Main grouping function

def print_organized_recipe(
    base_df,
    reagents_csv_path="master-reagents.csv",
    *,
    metal_ions=None,
    preferred_min_mass=0.1,
    fallback_min_mass=0.01,
    min_add_mL_per_L=0.01,
    max_mass_per_100mL=10.0,
    stock_volume_mL=100,
    max_trace_mass_per_1L=10.0,
    serial_dilutions=False
):
    """
    Print a user-friendly grouped preparation plan.

    Primary grouping logic is limited to:
    1. mass/stock-preparation constraints
    2. autoclave instruction

    Compatibility is retained as a constraint layer. Reagents are first considered
    within their mass/autoclave buckets, but they are only placed in the same
    subgroup if their ion/exclusion rules allow it.

    Reagents that fit the standard stock constraints are printed as BASE STOCKS.
    Reagents that do not fit are collected as TRACE STOCKS and printed as serial
    dilutions when serial_dilutions=True.

    `metal_ions` is retained for backward compatibility but is not used.
    """
    stock_volume_L = stock_volume_mL / 1000

    ions_map, non_autocl_map, excluded_map = _build_master_maps(reagents_csv_path)

    final_g = {
        str(r["reagent"]).strip(): float(r["g_per_L_final"])
        for _, r in base_df.iterrows()
        if float(r["g_per_L_final"]) > 0
    }

    items = []
    for reagent, g in final_g.items():
        ions = ions_map.get(reagent, set())
        excluded = excluded_map.get(reagent, set())
        items.append({
            "reagent": reagent,
            "g_per_L_final": g,
            "non_autocl": non_autocl_map.get(reagent, False),
            "ions": set(ions),
            "excluded": set(excluded),
        })

    unmeasurable = []

    print("\n== USER-FRIENDLY PREP ==\n")
    print("\n=== BASE STOCKS ===\n")

    roman_idx = 0

    for auto_bucket in [False, True]:
        bucket = [it for it in items if it["non_autocl"] == auto_bucket]
        bucket.sort(key=lambda x: x["g_per_L_final"], reverse=True)

        subgroups = []

        for it in bucket:
            placed = False
            for sg in subgroups:
                if not _can_place(it["ions"], it["excluded"], sg["_ions"], sg["_excluded"]):
                    continue

                test = sg["items"] + [it]
                factor, used_min = _find_stock_factor(
                    test,
                    final_g,
                    stock_volume_L=stock_volume_L,
                    preferred_min=preferred_min_mass,
                    fallback_min=fallback_min_mass,
                    min_add_mL=min_add_mL_per_L,
                    max_mass=max_mass_per_100mL,
                )

                if factor:
                    sg["items"].append(it)
                    sg["_ions"] |= it["ions"]
                    sg["_excluded"] |= it["excluded"]
                    sg["factor"] = factor
                    sg["used_min"] = used_min
                    placed = True
                    break

            if not placed:
                factor, used_min = _find_stock_factor(
                    [it],
                    final_g,
                    stock_volume_L=stock_volume_L,
                    preferred_min=preferred_min_mass,
                    fallback_min=fallback_min_mass,
                    min_add_mL=min_add_mL_per_L,
                    max_mass=max_mass_per_100mL,
                )

                if factor is None:
                    unmeasurable.append(it)
                else:
                    subgroups.append({
                        "items": [it],
                        "_ions": set(it["ions"]),
                        "_excluded": set(it["excluded"]),
                        "factor": factor,
                        "used_min": used_min,
                    })

        subgroups.sort(key=_group_sort_value, reverse=True)

        for sg in subgroups:
            sg["items"] = _items_desc(sg["items"])
            label = ROMAN[roman_idx] if roman_idx < len(ROMAN) else str(roman_idx + 1)
            roman_idx += 1
            factor = sg["factor"]
            add_mL = 1000.0 / factor

            header = (
                f"Group {label} — {factor}× base stock "
                f"(weigh into {stock_volume_mL} mL, add {_fmt_mL(add_mL)} per L)"
            )
            if auto_bucket:
                header = (
                    f"Group {label} — NON-AUTOCLAVABLE — {factor}× base stock "
                    f"(weigh into {stock_volume_mL} mL, add {_fmt_mL(add_mL)} per L)"
                )

            print(header + ":")
            for it in sg["items"]:
                mass = it["g_per_L_final"] * factor * stock_volume_L
                print(f"  {_fmt_mass(mass)} g {it['reagent']}")
            print()

    if serial_dilutions and unmeasurable:
        print("\n=== TRACE SERIAL STOCKS ===\n")
        roman_idx = 0

        for auto_bucket in [False, True]:
            bucket = [it for it in unmeasurable if it["non_autocl"] == auto_bucket]
            bucket.sort(key=lambda x: x["g_per_L_final"], reverse=True)
            if not bucket:
                continue

            compatible_groups = []
            for it in bucket:
                placed = False
                for sg in compatible_groups:
                    if not _can_place(it["ions"], it["excluded"], sg["_ions"], sg["_excluded"]):
                        continue
                    sg["items"].append(it)
                    sg["_ions"] |= it["ions"]
                    sg["_excluded"] |= it["excluded"]
                    placed = True
                    break

                if not placed:
                    compatible_groups.append({
                        "items": [it],
                        "_ions": set(it["ions"]),
                        "_excluded": set(it["excluded"]),
                    })

            compatible_groups.sort(key=_group_sort_value, reverse=True)

            for compat_group in compatible_groups:
                compat_group["items"] = _items_desc(compat_group["items"])
                split_sets = _split_trace_by_mass_window(
                    compat_group["items"],
                    preferred_min_mass=preferred_min_mass,
                    max_trace_mass=max_trace_mass_per_1L,
                )

                split_sets.sort(
                    key=lambda pair: max(x["g_per_L_final"] for x in pair[0]),
                    reverse=True,
                )

                for sub_items, primary_factor in split_sets:
                    sub_items = _items_desc(sub_items)
                    label = ROMAN[roman_idx] if roman_idx < len(ROMAN) else str(roman_idx + 1)
                    roman_idx += 1
                    working_factor = primary_factor // 1000

                    header = f"Group {label} — Compatible trace stock (serial dilution)"
                    if auto_bucket:
                        header = f"Group {label} — NON-AUTOCLAVABLE — Compatible trace stock (serial dilution)"

                    print(header)
                    print(f"  Step 1 — Primary stock ({int(primary_factor)}×, 1 L):")
                    print(f"    Weigh into 1 L DI water (target: <= {max_trace_mass_per_1L:g} g max reagent):")
                    for it in sub_items:
                        mass = it["g_per_L_final"] * primary_factor
                        print(f"      {_fmt_mass(mass)} g {it['reagent']}")

                    print(f"\n  Step 2 — Working stock ({int(working_factor)}×, 1 L):")
                    print("    Add 1 mL primary stock to 1 L DI water.")
                    print("\n  Step 3 — Media addition:")
                    print("    Add 1 mL working stock per 1 L media.\n")

    elif unmeasurable:
        print("\n=== TRACE STOCKS NEEDED ===\n")
        print("The following reagents could not meet the base-stock mass/pipetting constraints.")
        print("Run again with serial_dilutions=True to print trace serial stocks.\n")
        for it in sorted(unmeasurable, key=lambda x: x["g_per_L_final"], reverse=True):
            label = "NON-AUTOCLAVABLE" if it["non_autocl"] else "AUTOCLAVABLE"
            print(f"  {it['reagent']} ({label}): {it['g_per_L_final']:.6g} g/L final")
