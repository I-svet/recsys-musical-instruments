"""Выполнить experiments.ipynb текущим Python и сохранить выводы ячеек."""
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
cache = root / "artifacts" / "notebook"
cache.mkdir(parents=True, exist_ok=True)
os.environ["IPYTHONDIR"] = str(cache / "ipython")
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
os.environ["JUPYTER_RUNTIME_DIR"] = str(cache / "jupyter")
os.environ["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
path = root / "experiments.ipynb"
print("Чтение ноутбука", flush=True)
notebook = nbformat.read(path, as_version=4)
print("Запуск ядра Python", flush=True)
manager = KernelManager(kernel_name="python3", connection_file=str(cache / "kernel.json"))
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
print("Ядро настроено", flush=True)

def save(cell, cell_index, **kwargs):
    nbformat.write(notebook, path)
    print(f"Ячейка {cell_index + 1}/{len(notebook.cells)} выполнена", flush=True)

client = NotebookClient(notebook, km=manager, timeout=14400,
                        resources={"metadata": {"path": str(root)}},
                        on_cell_executed=save)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
if errors:
    raise RuntimeError("В ноутбуке есть ошибки")
print("Ноутбук выполнен и сохранён", flush=True)
