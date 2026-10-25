from pathlib import Path
import inspect

import nbformat

from recsys import io
from models import ItemKNN, ExplicitALS, EASE, FM, TimeSVDPP


def test_code_hash_covers_models_outside_recsys(tmp_path, monkeypatch):
    source = tmp_path / "hw1/src/recsys"
    models = tmp_path / "hw1/models"
    source.mkdir(parents=True)
    models.mkdir()
    (source / "io.py").write_text("helper = 1\n")
    model = models / "als.py"
    model.write_text("implementation = 1\n")
    monkeypatch.setattr(io, "__file__", str(source / "io.py"))
    before = io.code_hash()
    model.write_text("implementation = 2\n")
    assert io.code_hash() != before


def test_models_are_defined_in_separate_delivery_files():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(inspect.getfile(cls)).resolve() for cls in [ItemKNN, ExplicitALS, EASE, FM, TimeSVDPP]]
    assert len(set(paths)) == 5
    assert all(path.parent == root / "models" for path in paths)


def test_notebook_is_valid_and_fully_executed():
    notebook = nbformat.read(Path(__file__).resolve().parents[1] / "experiments.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert all(o.output_type != "error" for c in code for o in c.outputs)
    images = [o for c in code for o in c.outputs if "image/png" in o.get("data", {})]
    assert len(images) >= 4
