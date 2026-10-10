import pandas as pd
from pathlib import Path


def export_final_ion_concentrations(
    all_ions: list[str],
    ion_contributions: list | tuple | pd.Series,
    *,
    output_dir: str | Path = ".",
    filename: str = "final_ion_concentrations.csv",
):
    """
    Export final ion concentrations to a CSV file for downstream workflows.

    Parameters
    ----------
    all_ions : list of str
        List of ion names corresponding to each concentration value. Order is
        preserved in the output file.
    ion_contributions : array-like
        Final ion concentrations (mmol/L), typically representing the combined
        base and trace contributions.
    output_dir : str or pathlib.Path, optional
        Directory where the output CSV will be written. Created if it does not
        exist. Default is the current working directory.
    filename : str, optional
        Name of the output CSV file. Default is
        "final_ion_concentrations.csv".

    Returns
    -------
    pathlib.Path
        Full path to the written CSV file.

    Behavior
    --------
    - Constructs a two-column DataFrame with columns:
        'ion' and 'concentration'.
    - Ensures the output directory exists (creates it if necessary).
    - Writes the DataFrame to CSV without an index.
    - Prints the output file path upon successful export.

    Notes
    -----
    - Assumes `all_ions` and `ion_contributions` are aligned in length and order.
    - Concentrations are written as provided (no unit conversion or validation).
    - Intended for use as input to downstream formatting, speciation, or
      charge-balance routines.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "ion": all_ions,
        "concentration": ion_contributions,
    })

    outpath = output_dir / filename
    df.to_csv(outpath, index=False)

    print(f"✅ Final ion concentrations saved to: {outpath}")

    return outpath