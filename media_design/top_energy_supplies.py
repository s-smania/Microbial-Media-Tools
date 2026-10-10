import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def print_top_energy_supplies(
    energy_file,
    reaction_file,
    sample_name_col=None,
    top_n=10
):
    """
    Print the top energy-supplying reactions for each sample as HTML tables.

    Parameters
    ----------
    energy_file : str or pathlib.Path
        Path to the CSV file containing energy supply values by sample and
        reaction. Columns containing energy values must include the substring
        'energy supply_J/kg fluid'.
    reaction_file : str or pathlib.Path
        Path to the CSV file containing reaction metadata. This file must
        include:
        - 'reaction_name' : reaction identifier (e.g., 'rxn_1_1')
        - 'reaction' : HTML-formatted reaction string
    sample_name_col : str or None, optional
        Column name to use for sample labels in the output. If None, labels are
        generated as 'Sample {i}' based on row position. Default is None.
    top_n : int, optional
        Number of top energy-supplying reactions to display per sample.
        Default is 10.

    """

    energy_df = pd.read_csv(energy_file, sep=",")
    energy_df.columns = ["Sample"] + list(energy_df.columns[1:]) #fix formatting so it can call on samples
    
    rxn_df = pd.read_csv(reaction_file, sep=",")

    energy_cols = [
    col for col in energy_df.columns
    if "energy supply_J/kg fluid" in col
]

    energy_df[energy_cols] = energy_df[energy_cols].apply(
        pd.to_numeric, errors="coerce"
    )
   
 
    # Map energy column -> rxn ID (rxn_n_n)
    col_to_rxn = {
        col: re.search(r"(rxn_\d+_\d+)", col).group(1)
        for col in energy_cols
    }

    # Build lookup dictionary from reaction_name -> reaction (HTML)
    rxn_lookup = dict(zip(rxn_df["reaction_name"], rxn_df["reaction"]))

    # Loop through each sample
    for i, row in energy_df.iterrows():

        sample_label = (
            row[sample_name_col]
            if sample_name_col is not None
            else f"Sample {i}"
        )

        # Collect rxn values for this sample
        sample_pairs = [
            (rxn_id, row[col])
            for col, rxn_id in col_to_rxn.items()
            if pd.notna(row[col])
        ]

        # Sort descending by energy
        top_rxns = sorted(sample_pairs, key=lambda x: x[1], reverse=True)[:top_n]

        # Build HTML
        html = f"<h3>{sample_label}</h3>"
        html += "<table border='1' style='border-collapse: collapse;'>"
        html += "<tr><th>Reaction</th><th>Energy Supply (J/kg fluid)</th></tr>"

        for rxn_id, value in top_rxns:
            html += (
                "<tr>"
                f"<td>{rxn_lookup[rxn_id]}</td>"
                f"<td>{value:.3f}</td>"
                "</tr>"
            )

        html += "</table><br>"

        display(HTML(html))

def plot_top_energy_supplies(
    energy_file,
    reaction_file,
    sample_name_col="Sample",
    top_n=10
):
    """
    Plot the top energy-supplying reactions across samples as grouped bar charts.

    Parameters
    ----------
    energy_file : str or pathlib.Path
        Path to the CSV file containing energy supply values by sample and
        reaction. Columns containing energy values must include the substring
        'energy supply_J/kg fluid'.
    reaction_file : str or pathlib.Path
        Path to the CSV file containing reaction metadata. This file must
        include:
        - 'reaction_name' : reaction identifier (e.g., 'rxn_1_1')
        - 'reaction' : reaction string, potentially containing HTML formatting
    sample_name_col : str, optional
        Column name to use for sample labels in the plot legend. Default is
        'Sample'.
    top_n : int, optional
        Number of top energy-supplying reactions to retain per sample before
        taking the union across samples. Default is 10.

    Returns
    -------
    None
        This function displays a matplotlib figure and does not return a value.

    """

    energy_df = pd.read_csv(energy_file, sep=",")
    energy_df.columns = ["Sample"] + list(energy_df.columns[1:]) #fix formatting so it can call on samples
    rxn_df = pd.read_csv(reaction_file, sep=",")

    # Identify energy columns
    energy_cols = [
        col for col in energy_df.columns
        if "energy supply_J/kg fluid" in col
    ]

    # Convert to numeric
    energy_df[energy_cols] = energy_df[energy_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Map energy column -> rxn ID
    col_to_rxn = {
        col: re.search(r"(rxn_\d+_\d+)", col).group(1)
        for col in energy_cols
    }

    # Build lookup from reaction_name -> HTML reaction
    rxn_lookup = dict(zip(rxn_df["reaction_name"], rxn_df["reaction"]))

    # Strip HTML tags for plotting labels
    def strip_html(html_string):
        return re.sub("<.*?>", "", html_string)

    # Store top reactions per sample
    sample_top = {}

    for _, row in energy_df.iterrows():

        sample_label = row[sample_name_col]

        sample_pairs = [
            (rxn_id, row[col])
            for col, rxn_id in col_to_rxn.items()
            if pd.notna(row[col])
        ]

        top_rxns = sorted(
            sample_pairs,
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        sample_top[sample_label] = dict(top_rxns)

    # Union of all rxns appearing in any sample's top 10
    all_rxns = sorted(
        set(rxn for s in sample_top.values() for rxn in s.keys())
    )

    samples = list(sample_top.keys())

    # Build data matrix
    data_matrix = []

    for sample in samples:
        row = [
            sample_top[sample].get(rxn, 0)
            for rxn in all_rxns
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Remove reactions that are zero across all samples
    nonzero_mask = data_matrix.sum(axis=0) > 0
    data_matrix = data_matrix[:, nonzero_mask]
    all_rxns = [r for r, keep in zip(all_rxns, nonzero_mask) if keep]

    # Convert rxn IDs to readable reaction strings
    labels = [
        strip_html(rxn_lookup[rxn])
        for rxn in all_rxns
    ]

    # Plot
    x = np.arange(len(labels))
    width = 0.6 / len(samples)
    
    plt.figure(figsize=(10, 6))


    for i, sample in enumerate(samples):
        plt.bar(
            x + i * width,
            data_matrix[i],
            width=width,
            label=sample
        )

    plt.xticks(
        x + width * (len(samples) - 1) / 2,
        labels,
        rotation=60,
        ha="right",
        rotation_mode="anchor"
    )

    plt.ylabel("Energy Supply (J/kg fluid)")
    plt.xlabel("Reaction")
    plt.title("Top Energy-Supplying Reactions Per Sample")
    plt.legend()

    plt.tight_layout()
    plt.show()