"""
Execute the example notebooks in this directory end to end.

Each notebook is run in a temporary copy of the test directory so that the
files it writes (CSVs, saved speciation objects, figures) never touch the
repository. A failing cell fails the test.

Some media_design functions prompt the user with input(). Those prompts are
answered automatically by a setup cell injected at the top of each notebook.
"""

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

TEST_DIR = Path(__file__).resolve().parent
CELL_TIMEOUT = 1200  # seconds; speciation can be slow on CI runners

# Scripted answers for interactive prompts, matched by substring of the prompt
SETUP_CELL = r'''
import matplotlib
matplotlib.use("Agg")

_ANSWERS = {
    "RECIPE pH": "7.5",
    "RECIPE Temperature": "72",
    "(y/n)": "n",
}

def _scripted_input(prompt=""):
    for key, answer in _ANSWERS.items():
        if key in prompt:
            print(prompt + answer)
            return answer
    raise RuntimeError(f"No scripted answer for input prompt: {prompt!r}")

import media_design.adjust_ions
import media_design.speciation_formatter
media_design.adjust_ions.input = _scripted_input
media_design.speciation_formatter.input = _scripted_input
'''

NOTEBOOKS = sorted(TEST_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(notebook, tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(
        TEST_DIR, work_dir,
        ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"),
    )

    nb = nbformat.read(work_dir / notebook.name, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL))

    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    client.execute()
