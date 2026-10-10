import pandas as pd
import numpy as np
HCO3_MOLAR_MASS_G_PER_MOL = 61.016

# Helper functions
def _norm_str(x) -> str:
    """
    Normalize a value to a stripped string.

    Parameters
    ----------
    x : Any
        Input value to normalize.

    Returns
    -------
    str
        Empty string if `x` is null-like according to `pandas.isna`;
        otherwise `x` converted to string with surrounding whitespace removed.

    Notes
    -----
    This helper is used throughout the module to standardize comparisons
    against header names, ion names, and unit labels.
    """
    return "" if pd.isna(x) else str(x).strip()


def _is_unit(unit_value, target: str) -> bool:
    """
    Compare a unit label against a target string, case-insensitively.

    Parameters
    ----------
    unit_value : Any
        Unit value read from the input table.
    target : str
        Target unit string to compare against.

    Returns
    -------
    bool
        True if the normalized unit matches `target` ignoring case,
        otherwise False.
    """
    return _norm_str(unit_value).lower() == target.lower()


def _drop_excluded_ion_columns(
    df: pd.DataFrame,
    excluded,
) -> pd.DataFrame:
    """
    Remove columns whose ion header matches an excluded ion name.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame in speciation-style layout:
        row 0 contains ion headers, row 1 contains units, and row 2+
        contains sample data.
    excluded : iterable or None
        Collection of ion names to exclude. Matching is performed against
        row 0 values after string normalization.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with excluded ion columns removed. The original
        row structure is preserved.

    Notes
    -----
    - If `excluded` is None or empty, the input DataFrame is returned unchanged.
    - Matching is exact after whitespace stripping.
    """
    if not excluded:
        return df

    excluded_set = {str(x).strip() for x in excluded if str(x).strip()}
    if not excluded_set:
        return df

    keep_cols = []
    for c in df.columns:
        ion = _norm_str(df.loc[0, c])
        if ion in excluded_set:
            continue
        keep_cols.append(c)

    return df.loc[:, keep_cols]

def _slice_to_one_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    """
    Reduce a speciation-style table to a single sample.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with row 0 as headers, row 1 as units, and row 2+
        as sample data.
    sample : str
        Sample identifier to search for anywhere in a row.

    Returns
    -------
    pandas.DataFrame
        A three-row DataFrame containing:
        - row 0: ion headers
        - row 1: units
        - row 2: the first matching sample row

    Notes
    -----
    The returned DataFrame index is reset so that the selected sample row
    always becomes row 2.
    """
    sample_row = _find_sample_row(df, sample)
    df = df.iloc[[0, 1, sample_row]].reset_index(drop=True)
    return df

def _ensure_hco3_molality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that an HCO3- column exists and is expressed in Molality.

    Parameters
    ----------
    df : pandas.DataFrame
        Speciation-style DataFrame where row 0 contains ion headers,
        row 1 contains units, and row 2+ contains sample data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an HCO3- column present and standardized to
        Molality units.

    Behavior
    --------
    - If HCO3- is missing, a new HCO3- column is appended with unit
      'Molality' and zero-filled values.
    - If HCO3- exists and the unit is 'ppm', values are converted to
      Molality using `HCO3_MOLAR_MASS_G_PER_MOL`.
    - If HCO3- exists and values are missing or non-numeric, they are
      coerced to numeric and filled with 0.0.
    - If the HCO3- unit is blank, it is set to 'Molality'.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with HCO3- normalized.

    Notes
    -----
    The ppm-to-molality conversion assumes:
    - ppm ≈ mg/L
    - 1 L solution ≈ 1 kg water
    """
    # Find HCO3- column (exact match in ion header row)
    hco3_cols = [c for c in df.columns if _norm_str(df.loc[0, c]) == "HCO3-"]

    if not hco3_cols:
        new_c = (max(df.columns) + 1) if len(df.columns) else 0
        df[new_c] = 0
        df.loc[0, new_c] = "HCO3-"
        df.loc[1, new_c] = "Molality"
        if len(df.index) > 2:
            df.loc[2:, new_c] = 0
        return df

    c = hco3_cols[0]
    unit = df.loc[1, c]

    # Coerce numeric values (data rows only), NaN -> 0
    vals = pd.to_numeric(df.loc[2:, c], errors="coerce").fillna(0.0)

    if _is_unit(unit, "ppm"):
        # ppm ~ mg/L; assume 1 L ~ 1 kg water
        # molality (mol/kg) ≈ (mg/L * 1e-3 g/mg) / (g/mol)
        molality = (vals * 1e-3) / HCO3_MOLAR_MASS_G_PER_MOL
        df.loc[2:, c] = molality
        df.loc[1, c] = "Molality"
    else:
        df.loc[2:, c] = vals
        # Only set unit if blank (otherwise leave it)
        if _norm_str(unit) == "":
            df.loc[1, c] = "Molality"

    return df


def _find_sample_row(df: pd.DataFrame, sample: str) -> int:
    """
    Locate the first row containing a sample identifier.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to search.
    sample : str
        Sample string to search for. Matching is case-insensitive and checks
        whether the sample string appears anywhere in a row.

    Returns
    -------
    int
        Integer row index of the first matching row.

    """
    row_idx = df[
        df.apply(
            lambda r: r.astype(str).str.contains(str(sample), case=False, na=False).any(),
            axis=1,
        )
    ].index
    if len(row_idx) == 0:
        raise ValueError(f"Sample '{sample}' not found in file.")
    return int(row_idx[0])


def _extract_molality_cols_standard(df: pd.DataFrame) -> list[int]:
    """
    Identify columns labeled as Molality in a standard input table.

    Parameters
    ----------
    df : pandas.DataFrame
        Speciation-style DataFrame where row 1 contains unit labels.

    Returns
    -------
    list of int
        Column indices whose row 1 entry equals 'Molality' ignoring case.

    Notes
    -----
    This function is used when `format_input` is run without speciation/
    charge balancing.
    """
    return [int(i) for i in df.loc[1][df.loc[1].astype(str).str.lower() == "molality"].index]


def _extract_molality_cols_input(df: pd.DataFrame) -> list[int]:
    """
    Identify input molality columns from AqEquil speciation output.

    Parameters
    ----------
    df : pandas.DataFrame
        Speciation output DataFrame where row 0 contains headers and
        row 1 contains unit labels.

    Returns
    -------
    list of int
        Column indices satisfying both of the following:
        - row 1 unit is 'Molality'
        - row 0 header contains '(input)' or '_(input)'

    Notes
    -----
    This function is intended for extracting original input concentrations
    from speciation output tables rather than species-calculated columns.
    """
    cols = []
    for i in df.columns:
        header = _norm_str(df.loc[0, i]).lower()
        unit = _norm_str(df.loc[1, i]).lower()
        if unit != "molality":
            continue
        if "(input)" in header or "_(input)" in header:
            cols.append(int(i))
    return cols


def _clean_input_ion_name(header: str) -> str:
    """
    Remove AqEquil input suffixes from an ion header.

    Parameters
    ----------
    header : str
        Column header string.

    Returns
    -------
    str
        Cleaned ion name with '(input)' and '_(input)' removed, trailing
        underscores stripped, and surrounding whitespace removed.
    """
    h = _norm_str(header)
    h = h.replace("_(input)", "").replace("(input)", "")
    return h.strip().rstrip("_").strip()
    

def _find_log_activity_col(df: pd.DataFrame, ion_name: str) -> int | None:
    """
    Find the log_activity column corresponding to a specified ion.

    Parameters
    ----------
    df : pandas.DataFrame
        Speciation output DataFrame where row 0 contains headers and
        row 1 contains unit labels.
    ion_name : str
        Ion name to locate.

    Returns
    -------
    int or None
        Column index of the first matching `log_activity` column, or None
        if no match is found.

    Notes
    -----
    Header matching is performed after removing AqEquil input suffixes
    using `_clean_input_ion_name`.
    """
    target = str(ion_name).strip()

    for c in df.columns:
        hdr = _clean_input_ion_name(df.loc[0, c])
        unit = _norm_str(df.loc[1, c]).lower()
        if hdr == target and unit == "log_activity":
            return int(c)
    return None


def format_input(
    input_file,
    sample,
    *,
    excluded_target_ions=None,
    charge_balance=False,
    charge_balance_ion="HCO3-",
):
    """
    Format a geochemical/speciation-style CSV into a standardized ion table.

    Parameters
    ----------
    input_file : str or path-like
        Path to the input CSV file. A bare file name that is not found in the
        current working directory but that matches a file bundled in the
        package 'data' directory resolves to the bundled copy. The file is
        expected to follow the speciation-style layout:
        - row 0: ion headers
        - row 1: units
        - row 2+: sample data
    sample : str
        Sample identifier to extract. The first row containing this string
        anywhere is selected.
    excluded_target_ions : iterable of str or None, optional
        Ion names to exclude from the formatted output. Matching is exact
        after whitespace stripping. Default is None.
    charge_balance : bool, optional
        If True, run AqEquil speciation and charge balancing on the selected
        sample before formatting output. Default is False.
    charge_balance_ion : str, optional
        Ion to use for AqEquil charge balancing. Default is "HCO3-".

    Returns
    -------
    tuple
        A tuple containing:
        
        - formatted_df : pandas.DataFrame
            Two-column DataFrame with:
            - 'ion': ion name
            - 'concentration': numeric concentration
        - geochemical_data : dict
            Dictionary mapping ion names to concentrations.

    Output
    ------
    formatted_df contains:
    - ion : str
    - concentration : float

    geochemical_data contains:
    - {ion_name: concentration}

    Notes
    -----
    - After sample slicing, the selected sample row is always row index 2.
    - When `charge_balance=False`, molality values are taken directly from
      the input file.
    - When `charge_balance=True`, molality values are extracted from AqEquil
      output columns marked as input values.
    - For the charge-balance ion, activity is assumed equal to molality
      after converting from `log_activity`.
    - HCO3- receives special handling to ensure explicit zero values are
      retained.
    - Excluded ions are filtered both before and after charge balancing in
      case the speciation output regenerates them.

    """
    import os
    import tempfile
    import pandas as pd

    from .paths import resolve_data_path

    df = pd.read_csv(resolve_data_path(input_file), header=None, dtype=object)

    # Drop excluded ion columns
    df = _drop_excluded_ion_columns(df, excluded_target_ions)

    # Slice to ONLY the sample of interest
    df = _slice_to_one_sample(df, sample)

    # HCO3 handling BEFORE any speciation/charge-balance
    df = _ensure_hco3_molality(df)

    # Optional speciation + charge balance
    if charge_balance:
        import aqequil

        with tempfile.TemporaryDirectory() as td:
            cwd0 = os.getcwd()
            try:
                os.chdir(td)

                df.to_csv("output_df.csv", header=False, index=False)

                ae = aqequil.AqEquil(db="WORM")
                ae.speciate(
                    input_filename="output_df.csv",
                    exclude=["Name", "Year"],
                    report_filename="speciation_output",
                    delete_generated_folders=True,
                    charge_balance_on=charge_balance_ion,
                )

                df = pd.read_csv("speciation_output.csv", header=None, dtype=object)

            finally:
                os.chdir(cwd0)

        molality_cols = _extract_molality_cols_input(df)
        ions = [_clean_input_ion_name(df.loc[0, i]) for i in molality_cols]

        # after slicing, the sample row is always row 2
        row_idx = 2

    else:
        molality_cols = _extract_molality_cols_standard(df)
        ions = [_norm_str(df.loc[0, i]) for i in molality_cols]
        row_idx = 2  # sample row is always row 2 after slicing

    values = [df.loc[row_idx, i] for i in molality_cols]

    formatted_df = pd.DataFrame(
        {"ion": ions, "concentration": pd.to_numeric(values, errors="coerce")}
    )

    # Special handling for the charge-balance ion: use log_activity to compute activity = 10^log_activity
    log_act_col = _find_log_activity_col(df, charge_balance_ion)
    if log_act_col is not None:
        log_act = pd.to_numeric(df.loc[row_idx, log_act_col], errors="coerce")
        if pd.isna(log_act):
            activity_m = 0.0
        else:
            activity_m = float(10 ** log_act)  # activity assumed = molality
    
        # Insert or override the charge-balance ion concentration
        mask = formatted_df["ion"].astype(str).str.strip() == str(charge_balance_ion).strip()
        if mask.any():
            formatted_df.loc[mask, "concentration"] = activity_m
        else:
            formatted_df = pd.concat(
                [formatted_df, pd.DataFrame({"ion": [charge_balance_ion], "concentration": [activity_m]})],
                ignore_index=True
            )
    
    # HCO3-: NaN -> 0 (and keep explicit zeros as zero)
    is_hco3 = formatted_df["ion"].astype(str).str.strip() == "HCO3-"
    formatted_df.loc[is_hco3, "concentration"] = formatted_df.loc[is_hco3, "concentration"].fillna(0.0)

    # drop remaining NaNs
    formatted_df = formatted_df.dropna(subset=["concentration"]).reset_index(drop=True)

    # safety: apply exclusion again (in case speciation regenerates them)
    if excluded_target_ions:
        excluded_set = {str(x).strip() for x in excluded_target_ions if str(x).strip()}
        formatted_df = formatted_df[~formatted_df["ion"].isin(excluded_set)].reset_index(drop=True)

    geochemical_data = dict(zip(formatted_df["ion"], formatted_df["concentration"]))

    print("\n✅ Formatted DataFrame (after exclusions):")
    print(formatted_df)

    return formatted_df, geochemical_data