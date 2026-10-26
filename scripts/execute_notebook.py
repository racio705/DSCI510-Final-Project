"""Generate, execute and export the saved-results notebook with this interpreter."""
from pathlib import Path
import sys
import os

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

from create_notebook import create_notebook

root = Path(__file__).resolve().parents[1]
for variable, subfolder in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter")]:
    directory = root / ".runtime" / subfolder
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
path = create_notebook(root)
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
NotebookClient(notebook, km=manager, timeout=120,
               resources={"metadata": {"path": str(root)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
exporter = HTMLExporter()
exporter.exclude_input = True
html, _ = exporter.from_notebook_node(notebook)
destination = root / "results/portfolio_results.html"
destination.write_text(html, encoding="utf-8")
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} cells: {path}")
print(destination)
