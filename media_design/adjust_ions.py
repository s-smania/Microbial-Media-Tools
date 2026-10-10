import pandas as pd

def adjust_ion_concentrations(
    input_csv="final_ion_concentrations.csv",
    output_csv=None
):
    """
    Interactively modify ion concentrations from a CSV file.
    
    Parameters
    ----------
    input_csv : str, optional
        Path to the input CSV file containing ion data. Default is
        "final_ion_concentrations.csv". The file must contain two columns:
        'ion' (str) and 'concentration' (float).
    
    output_csv : str or None, optional
        Path to write the updated CSV file. If None, the input file is
        overwritten. Default is None.
    
    Returns
    -------
    pandas.DataFrame or None
        Updated DataFrame containing modified ion concentrations if changes
        were made. Returns None if the user opts not to modify any values.
    
    Behavior
    --------
    - Prompts the user to confirm whether they want to adjust concentrations.
    - If 'n', exits without reading or writing any files.
    - If 'y', loads the input CSV and allows iterative modification of ion
        concentrations via terminal prompts.
    - Validates that selected ions exist in the dataset.
    - Re-prompts on invalid ion names or non-numeric inputs.
    - Allows repeated adjustments until the user exits.
    - Writes updated data to `output_csv` (or overwrites `input_csv` if None).
    
    Notes
    -----
    - Ion matching is case-sensitive and must exactly match entries in the
          'ion' column.
    - Only one concentration value per ion is assumed.

    """

    response = input("Do you want to change the concentrations of any ions? (y/n): ").strip().lower()

    if response != "y":
        print("No changes made.")
        return None  # signals: skip modification

    # Only load file if user wants changes
    df = pd.read_csv(input_csv)

    if "ion" not in df.columns or "concentration" not in df.columns:
        raise ValueError("CSV must contain columns: 'ion' and 'concentration'")

    while True:

        ion = input("What ion do you want to adjust?: ").strip()

        if ion not in df["ion"].values:
            print(f"Ion '{ion}' not found.")
            print("Available ions:")
            print(", ".join(df["ion"].values))
            continue

        current_value = df.loc[df["ion"] == ion, "concentration"].values[0]
        print(f"Your current concentration of {ion} is {current_value}")

        try:
            new_value = float(input("What would you like to change it to?: ").strip())
        except ValueError:
            print("Invalid number. Try again.")
            continue

        df.loc[df["ion"] == ion, "concentration"] = new_value
        print(f"{ion} updated from {current_value} → {new_value}")

        again = input("Would you like to adjust another ion? (y/n): ").strip().lower()
        if again != "y":
            break

    # Determine output file
    if output_csv is None:
        output_csv = input_csv  # overwrite default

    df.to_csv(output_csv, index=False)

    print(f"\nUpdated concentrations exported to: {output_csv}")

    return df