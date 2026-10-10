import pandas as pd

from .paths import data_path, resolve_data_path

REQUIRED_COLUMNS = {"reagent_name", "ion", "stoichiometry", "molar_mass"}

def _build_reagents_from_csv(reagents_csv_path: str) -> dict:
    """
    Build a reagent definition dictionary from a reagent CSV file.

    Parameters
    ----------
    reagents_csv_path : str
        Path to the reagent definition CSV file. The file must contain the
        following columns:
        - 'reagent_name' : reagent or compound name
        - 'ion' : ion contributed by the reagent
        - 'stoichiometry' : moles of ion contributed per mole of reagent
        - 'molar_mass' : reagent molar mass in g/mol

    Returns
    -------
    dict
        Nested reagent dictionary formatted as:

        .. code-block:: python

            {
                "NaCl": {
                    "ions": {"Na+": 1, "Cl-": 1},
                    "molar_mass": 58.44
                },
                "MgSO4·7H2O": {
                    "ions": {"Mg2+": 1, "SO4^2-": 1},
                    "molar_mass": 246.47
                }
            }

    Notes
    -----
    - Each reagent may appear across multiple rows in the CSV, one row per ion.
    - The returned dictionary stores one molar mass per reagent and a nested
      'ions' dictionary of stoichiometric contributions.
    - This function performs validation only on required structure and numeric
      fields; it does not validate chemical consistency.
    """
    
    df = pd.read_csv(reagents_csv_path)

    # Sanity checks
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{reagents_csv_path}' is missing required columns: {', '.join(sorted(missing))}."
        )

    # Type checks
    df["molar_mass"] = pd.to_numeric(df["molar_mass"], errors="coerce")
    df["stoichiometry"] = pd.to_numeric(df["stoichiometry"], errors="coerce")

    if df["molar_mass"].isna().any():
        raise ValueError(f"CSV '{reagents_csv_path}' contains non-numeric or missing molar_mass values.")
    if df["stoichiometry"].isna().any():
        raise ValueError(f"CSV '{reagents_csv_path}' contains non-numeric or missing stoichiometry values.")

    # Build reagent dictionary
    reagents = {
        name: {"ions": {}, "molar_mass": mass}
        for name, mass in df[["reagent_name", "molar_mass"]].drop_duplicates().values
    }

    df.apply(
        lambda row: reagents[row["reagent_name"]]["ions"].update(
            {row["ion"]: row["stoichiometry"]}
        ),
        axis=1,
    )

    return reagents


def load_reagents(reagents_csv_path: str | None = None,
                  excluded_reagents: list | set | None = None) -> dict:
    """
    Load reagent definitions from a user-provided CSV or the default reagent database.

    Parameters
    ----------
    reagents_csv_path : str or None, optional
        Path to a user-provided reagent definition CSV. If provided, the
        function attempts to load and validate this file first. If loading
        fails for any reason, the function falls back to the default
        'master-reagents.csv' file bundled in the package 'data' directory.
        A bare file name that is not found in the current working directory
        but that matches a bundled data file resolves to the bundled copy.
        If None, the default file is used directly.
    excluded_reagents : list or set or None, optional
        Iterable of reagent names to exclude from the returned reagent
        dictionary. Matching is case-sensitive. Default is None.

    Returns
    -------
    dict
        Nested reagent dictionary formatted as:

        .. code-block:: python

            {
                "NaCl": {
                    "ions": {"Na+": 1, "Cl-": 1},
                    "molar_mass": 58.44
                },
                "MgSO4·7H2O": {
                    "ions": {"Mg2+": 1, "SO4^2-": 1},
                    "molar_mass": 246.47
                }
            }

    Notes
    -----
    - Fallback behavior is intentional: invalid user CSV files do not stop
      execution if the default reagent database is available.
    - Exclusions are applied after loading, regardless of whether the source
      was the user CSV or the default CSV.
    - Printed messages document which source file was used and how many
      reagent entries were retained.
    """

    # Default CSV lives in the package data directory
    default_csv = data_path("master-reagents.csv")

    # Try user-provided CSV (bare names fall back to the bundled copy)
    if reagents_csv_path:
        reagents_csv_path = resolve_data_path(reagents_csv_path)
        try:
            reagents = _build_reagents_from_csv(reagents_csv_path)
            print(f"Loaded reagent definitions from your CSV: {reagents_csv_path}")
        except Exception as e:
            print(f"⚠️ Your CSV '{reagents_csv_path}' could not be used: {e}")
            print(f"➡ Falling back to default CSV: {default_csv}")
            reagents = _build_reagents_from_csv(default_csv)
            print(f"Loaded reagent definitions from default CSV: {default_csv}")
    else:
        reagents = _build_reagents_from_csv(default_csv)
        print(f"Loaded reagent definitions from default CSV: {default_csv}")

    # Apply exclusions
    if excluded_reagents:
        excluded_reagents = set(excluded_reagents)
        reagents = {n: d for n, d in reagents.items() if n not in excluded_reagents}
        print(f"Excluded reagents: {sorted(excluded_reagents)}")

    print(f"✅ Reagents loaded: {len(reagents)} entries.")
    return reagents


def check_ion_coverage(reagents: dict, geochemical_data: dict):
    """
    Verify that all target ions are represented in the loaded reagent set.

    Parameters
    ----------
    reagents : dict
        Reagent definition dictionary as returned by `load_reagents` or
        `_build_reagents_from_csv`. Each reagent entry must include an
        'ions' sub-dictionary mapping ion names to stoichiometric
        coefficients.
    geochemical_data : dict
        Dictionary mapping target ion names to target concentrations.
        Typically generated from formatted input data or speciation output.

    Returns
    -------
    None
        This function does not return a value. It raises an exception if
        coverage is incomplete.

    Notes
    -----
    - Coverage is evaluated only on ion names, not concentrations.
    - Matching is exact and case-sensitive.
    - This function is intended as a pre-check before optimization or recipe
      generation steps.
    """
    all_ions = set()
    for reagent in reagents.values():
        all_ions.update(reagent["ions"].keys())
    all_ions = list(all_ions)

    target_ions = list(geochemical_data.keys())
    missing_ions = [ion for ion in target_ions if ion not in all_ions]

    if missing_ions:
        print("\n[ERROR] Missing ion coverage for:", ", ".join(missing_ions))
        raise ValueError(f"Missing ion coverage for: {', '.join(missing_ions)}")

    print("✅ All target ions covered by reagent set.")