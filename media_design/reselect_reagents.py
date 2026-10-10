import os
import pandas as pd

from .paths import resolve_data_path


def select_reagents(
    type="custom",
    reagent_csv_path=None,
    master_reagents_path=None,
    custom_reagents_path=None,
    reagents=None,
):
    """
    Select the reagent dataset to use for recipe construction.

    Parameters
    ----------
    type : {"custom", "preexist"}, optional
        Reagent selection mode. Supported options are:
        - "custom": use an already-loaded reagent DataFrame
        - "preexist": subset a reagent database based on reagents present in a
          preexisting recipe CSV
        Default is "custom".
    reagent_csv_path : str or None, optional
        Path to a preexisting recipe CSV file. Required when `type="preexist"`.
        The file must contain a 'reagent' column listing reagent names used in
        the recipe. Default is None.
    master_reagents_path : str or None, optional
        Path to the default/master reagent database CSV used in
        `type="preexist"` mode when `custom_reagents_path` is not provided.
        The file must contain a 'reagent' column. If None, the
        'master-reagents.csv' file bundled in the package 'data' directory is
        used. Default is None.
    custom_reagents_path : str or None, optional
        Path to a user-supplied reagent database CSV to use instead of the
        master reagent database in `type="preexist"` mode. The file must
        contain a 'reagent' column. Default is None.
    reagents : pandas.DataFrame or None, optional
        Already-loaded reagent DataFrame used in `type="custom"` mode. This
        DataFrame is returned as a copy. Default is None.

    Returns
    -------
    pandas.DataFrame
        Reagent DataFrame to use for downstream recipe construction.

        - In `type="custom"` mode, returns a copy of the provided `reagents`
          DataFrame.
        - In `type="preexist"` mode, returns the subset of the selected reagent
          database whose reagent names match those found in the recipe file.

    Notes
    -----
    - Matching is based on exact reagent-name equality after whitespace
      stripping.
    - In `type="preexist"` mode, `custom_reagents_path` takes precedence over
      `master_reagents_path`.
    - Missing reagents from the recipe are printed for visibility, but the
      function only raises an error if no matches are found at all.
    - This function returns reagent rows only; it does not validate the
      chemical completeness or stoichiometric consistency of the selected set.

    """

    if type not in ["custom", "preexist"]:
        raise ValueError('type must be "custom" or "preexist"')

    
    # Custom reagents mode
    
    if type == "custom":
        if reagents is None:
            raise ValueError(
                "For type='custom', you must provide the already-loaded reagents DataFrame."
            )

        print("\nReagent source: previously loaded reagents (custom recipe workflow)")
        print(f"Number of reagents available: {len(reagents)}")

        return reagents.copy()

    
    # PREEXISTING RECIPE MODE
    
    if reagent_csv_path is None:
        raise ValueError("reagent_csv_path is required when type='preexist'")

    recipe_df = pd.read_csv(resolve_data_path(reagent_csv_path))

    if "reagent" not in recipe_df.columns:
        raise ValueError("Input recipe CSV must contain a 'reagent' column")

    recipe_reagents = set(recipe_df["reagent"].astype(str).str.strip())

    print("\nReagent selection mode: preexisting recipe")
    print(f"Recipe file: {reagent_csv_path}")
    print(f"Reagents detected in recipe: {len(recipe_reagents)}")

    # -----------------------------------------
    # USER-SUPPLIED REAGENT FILE
    # -----------------------------------------
    if custom_reagents_path is not None:
        print(f"\nUsing user-supplied reagent file: {custom_reagents_path}")
        master = pd.read_csv(resolve_data_path(custom_reagents_path))

    # -----------------------------------------
    # MASTER REAGENTS
    # -----------------------------------------
    else:
        master_path = resolve_data_path(master_reagents_path,
                                        default="master-reagents.csv")

        print(f"\nUsing master reagents file: {master_path}")
        master = pd.read_csv(master_path)

    if "reagent" not in master.columns:
        raise ValueError("Reagents file must contain a 'reagent' column")

    master["reagent"] = master["reagent"].astype(str).str.strip()

    subset = master[master["reagent"].isin(recipe_reagents)].copy()

    print(f"Matched reagents in database: {len(subset)}")

    missing = recipe_reagents - set(master["reagent"])

    if missing:
        print("\nReagents in recipe not found in reagent database:")
        for r in sorted(missing):
            print(f"  - {r}")

    if subset.empty:
        raise ValueError(
            "No reagents from the recipe matched the reagent database."
        )

    return subset