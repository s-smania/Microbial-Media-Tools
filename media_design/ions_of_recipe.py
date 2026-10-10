from pathlib import Path
import pandas as pd
import numpy as np

from .paths import data_path, resolve_data_path


def convert_reagents_to_ions(
    reagent_csv_path,
    output_csv_path,
    pH,
    temperature,
    pressure,
    master_reagents_path=""
):
    """
    Convert reagent concentrations to ion molalities formatted for speciation input.

    Parameters
    ----------
    reagent_csv_path : str or pathlib.Path
        Path to a CSV file containing reagent names and concentrations. The file
        is expected to contain two columns that will be interpreted as
        'reagent' and 'concentration'. The first row of the concentration
        column must contain the unit string, which must include one of:
        'mg', 'g', or 'mol' (for mg/L, g/L, or mol/L input).
    output_csv_path : str or pathlib.Path
        Path where the formatted speciation CSV will be written.
    pH : float
        pH value to insert into the output file under the 'H+' column if that
        column exists in the speciation template.
    temperature : float
        Temperature value to insert into the output file under the
        'Temperature' column if that column exists in the speciation template.
    pressure : float
        Pressure value to insert into the output file under the 'Pressure'
        column if that column exists in the speciation template.
    master_reagents_path : str or pathlib.Path, optional
        Path to the master reagent database CSV. If not provided, the function
        uses 'master-reagents.csv' bundled in the package 'data' directory.

    Returns
    -------
    pandas.DataFrame
        A formatted DataFrame containing:
        - row 0: unit labels
        - row 1: the generated recipe/sample data row

    Input Format
    ------------
    The reagent input CSV must contain two columns representing:
    - reagent
    - concentration

    The first row of the concentration column is interpreted as the unit label,
    not as data. All remaining rows are treated as reagent concentration data.

    Supported unit conventions:
    - mg/L
    - g/L
    - mol/L

    Notes
    -----
    - Reagent concentrations are converted as follows:
        - mg/L to mol/L using:
          (concentration / 1000) / molar_mass
        - g/L to mol/L using:
          concentration / molar_mass
        - mol/L is used directly
    - Ion molality is calculated as:
      reagent mol/L × stoichiometric coefficient
    - Ion totals are summed across all contributing reagents.
    - Missing ions are appended to the speciation template and
      assigned the unit label 'Molality'.

    """

    # Resolve internal paths
    master_path = resolve_data_path(master_reagents_path, default="master-reagents.csv")
    speciation_path = data_path("speciation-formatting.csv")


    # Load reagent file
    reagent_csv_path = Path(reagent_csv_path)
    file_name = reagent_csv_path.stem

    reag_df = pd.read_csv(reagent_csv_path)
    reag_df.columns = ["reagent", "concentration"]

    unit_string = str(reag_df.iloc[0]["concentration"]).strip().lower()
    reag_df = reag_df.iloc[1:].copy()

    reag_df["concentration"] = pd.to_numeric(
        reag_df["concentration"], errors="coerce"
    ).fillna(0.0)


    # Load master reagents
    master = pd.read_csv(master_path)
    master["stoichiometry"] = pd.to_numeric(master["stoichiometry"], errors="coerce")
    master["molar_mass"] = pd.to_numeric(master["molar_mass"], errors="coerce")

    mm = (
        master[["reagent_name", "molar_mass"]]
        .drop_duplicates()
        .set_index("reagent_name")["molar_mass"]
    )

    reag_df["molar_mass"] = reag_df["reagent"].map(mm)

    if reag_df["molar_mass"].isna().any():
        missing = reag_df.loc[reag_df["molar_mass"].isna(), "reagent"].tolist()
        raise ValueError(f"Reagents not found in master-reagents: {missing}")

    # Convert to mol/L
    if "mg" in unit_string:
        reag_df["mol_L"] = (reag_df["concentration"] / 1000.0) / reag_df["molar_mass"]

    elif "g" in unit_string:
        reag_df["mol_L"] = reag_df["concentration"] / reag_df["molar_mass"]

    elif "mol" in unit_string:
        reag_df["mol_L"] = reag_df["concentration"]

    else:
        raise ValueError(
            f"Unsupported units detected: '{unit_string}'. "
            "Must contain mg/L, g/L, or mol/L."
        )

    # Apply stoichiometry
    merged = master.merge(
        reag_df[["reagent", "mol_L"]],
        left_on="reagent_name",
        right_on="reagent",
        how="inner",
    )

    merged["ion_molality"] = merged["mol_L"] * merged["stoichiometry"]

    ion_totals = (
        merged.groupby("ion")["ion_molality"]
        .sum()
        .sort_index()
    )


    # Load speciation template
    spec_template = pd.read_csv(speciation_path, header=None, dtype=object)

    header_row = spec_template.iloc[0].tolist()
    units_row = spec_template.iloc[1].tolist()


    # Add missing ions dynamically
    for ion in ion_totals.index:
        if ion not in header_row:
            header_row.append(ion)
            units_row.append("Molality")


    # Build output row
    output_data = {col: "" for col in header_row}

    # Force Sample and Name columns
    if "Sample" in header_row:
        output_data["Sample"] = "recipe"

    if "Name" in header_row:
        output_data["Name"] = file_name

    # Insert environmental parameters
    if "H+" in header_row:
        output_data["H+"] = pH

    if "Temperature" in header_row:
        output_data["Temperature"] = temperature

    if "Pressure" in header_row:
        output_data["Pressure"] = pressure

    # Fill ion molalities
    for ion, value in ion_totals.items():
        output_data[ion] = float(value)
        idx = header_row.index(ion)
        units_row[idx] = "Molality"


    # Construct final dataframe
    final_df = pd.DataFrame(
        [units_row, [output_data[col] for col in header_row]],
        columns=header_row,
    )

    final_df.to_csv(output_csv_path, index=False)

    return final_df