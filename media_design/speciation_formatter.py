import pandas as pd

from .paths import resolve_data_path


def build_speciation_recipe_file(
    sample: str,
    fmt_path: str,
    op_path: str,
    final_path: str,
    *,
    out_path: str | None = None,
    H_plus_recipe: float | None = None,
    Temperature_recipe: float | None = None,
) -> str:
    """
    Build a speciation-ready CSV by combining a formatting template, original sample data,
    and a final recipe ion table.

    Parameters
    ----------
    sample : str
        Sample identifier to locate in `op_path`. The first row containing this
        string anywhere is used as the original sample row.
    fmt_path : str
        Path to the speciation-formatting template CSV. A bare file name that
        is not found in the current working directory but that matches a file
        bundled in the package 'data' directory resolves to the bundled copy.
        This file is read with `header=None` and is expected to contain:
        - row 0: column headers
        - row 1: unit labels
    op_path : str
        Path to the original/speciation input CSV containing the source sample
        data. This file is read with `header=None` and must contain the sample
        specified by `sample`.
    final_path : str
        Path to the final ion concentration CSV. This file must contain:
        - 'ion'
        - 'concentration'
    out_path : str or None, optional
        Path for the output CSV. If None, the file is written as
        `formatted__{sample}.csv`. Default is None.
    H_plus_recipe : float or None, optional
        Recipe pH value to write into the 'H+' column of the recipe row. If
        None, the user is prompted interactively. Default is None.
    Temperature_recipe : float or None, optional
        Recipe temperature in °C to write into the 'Temperature' column of the
        recipe row. If None, the user is prompted interactively. Default is
        None.

    Returns
    -------
    str
        Path to the written output CSV file.

    Notes
    -----
    - The 'Pressure' column is always removed from the formatting template
      before output generation
    - Missing ions are appended only if they are present in both:
      - the final ion table (`final_path`)
      - the original sample file (`op_path`)
      but absent from the formatting template.
    - Ion columns are identified only where the unit row is 'Molality' or
      'ppm', case-insensitive.
    - Recipe HCO3- values are converted from molality to ppm using:
      `molality * 61.016 * 1000`
    - The first matching sample row in `op_path` is used if multiple rows
      contain the sample string.
    - Output is written with `header=False` and `index=False`.

    """


    # Prompt for recipe conditions

    if H_plus_recipe is None:
        H_plus_recipe = float(input("Enter RECIPE pH (to store in H+ column): "))

    if Temperature_recipe is None:
        Temperature_recipe = float(input("Enter RECIPE Temperature (°C): "))


    # Load files

    fmt = pd.read_csv(resolve_data_path(fmt_path, default="speciation-formatting.csv"),
                      header=None, dtype=object)
    op = pd.read_csv(resolve_data_path(op_path), header=None, dtype=object)
    final = pd.read_csv(final_path)


    # Remove pressure column

    header_row = fmt.loc[0].astype(str)
    pressure_cols = header_row[header_row.str.strip().str.lower() == "pressure"].index.tolist()

    if pressure_cols:
        fmt = fmt.drop(columns=pressure_cols)


    # Identify molality/ppm columns

    fmt_second_row = fmt.loc[1].astype(str).str.strip().str.lower()
    fmt_mol_cols = fmt_second_row[fmt_second_row.isin(["molality", "ppm"])].index.tolist()
    fmt_ions = fmt.loc[0, fmt_mol_cols].astype(str).tolist()

    op_second_row = op.loc[1].astype(str).str.strip().str.lower()
    op_mol_cols = op_second_row[op_second_row.isin(["molality", "ppm"])].index.tolist()
    op_ions = op.loc[0, op_mol_cols].astype(str).tolist()

    final_ions = final["ion"].astype(str).tolist()


    # Append missing ions

    missing_ions = [
        ion for ion in final_ions
        if ion in op_ions and ion not in fmt_ions
    ]

    if missing_ions:
        print("\nAppending new ion columns:")
        for ion in missing_ions:
            print(" -", ion)

            new_col = fmt.shape[1]
            fmt[new_col] = ""
            fmt.loc[0, new_col] = ion
            fmt.loc[1, new_col] = "Molality"

        # Recalculate mol columns
        fmt_second_row = fmt.loc[1].astype(str).str.strip().str.lower()
        fmt_mol_cols = fmt_second_row[fmt_second_row.isin(["molality", "ppm"])].index.tolist()
        fmt_ions = fmt.loc[0, fmt_mol_cols].astype(str).tolist()


    # Locate sample row in op_path

    sidx_list = op[
        op.apply(
            lambda r: r.astype(str).str.contains(sample, case=False, na=False).any(),
            axis=1,
        )
    ].index.tolist()

    if not sidx_list:
        raise ValueError(f"Sample '{sample}' not found in {op_path}")

    sidx = sidx_list[0]


    # Helper: find column index

    def col_idx(name: str):
        cols = fmt.loc[0].astype(str)
        matches = cols[cols == name].index.tolist()
        return matches[0] if matches else None

    def get_meta(key: str):
        ci = col_idx(key)
        if ci is None:
            return ""
        op_cols = op.loc[0].astype(str)
        op_match = op_cols[op_cols == key].index.tolist()
        if not op_match:
            return ""
        return op.loc[sidx, op_match[0]]


    # Build recipe row

    row_recipe = pd.Series("", index=fmt.columns, dtype=object)

    for k in ["Sample", "Name", "Year"]:
        ci = col_idx(k)
        if ci is not None:
            val = get_meta(k)
            if k == "Sample":
                val = "recipe"
            row_recipe[ci] = val

    for k, v in {
        "H+": H_plus_recipe,
        "Temperature": Temperature_recipe,
    }.items():
        ci = col_idx(k)
        if ci is not None:
            row_recipe[ci] = v

    final_map = dict(
        zip(
            final["ion"].astype(str),
            pd.to_numeric(final["concentration"], errors="coerce"),
        )
    )

    for col, ion in zip(fmt_mol_cols, fmt_ions):
        if ion in final_map and pd.notnull(final_map[ion]):
            val = final_map[ion]
    
            # convert recipe HCO3- from molality → ppm
            if ion == "HCO3-":
                val = float(val) * 61.016 * 1000
    
            row_recipe[col] = val


    # Build original sample data row

    row_orig = pd.Series("", index=fmt.columns, dtype=object)

    for k in ["Sample", "Name", "Year", "H+", "Temperature"]:
        ci = col_idx(k)
        if ci is not None:
            row_orig[ci] = get_meta(k)

    op_header = op.loc[0].astype(str)
    op_map = {ion: i for i, ion in op_header.items()}
    
    for col, ion in zip(fmt_mol_cols, fmt_ions):
        if ion in op_map:
            row_orig[col] = op.loc[sidx, op_map[ion]]


    # Combine and export

    out = pd.concat(
        [fmt, row_recipe.to_frame().T, row_orig.to_frame().T],
        ignore_index=True,
    )

    if out_path is None:
        out_path = f"formatted__{sample}.csv"

    out.to_csv(out_path, index=False, header=False)

    print(f"\nWrote: {out_path}")

    return out_path