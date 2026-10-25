"""Explicit small registry keeps all model implementations inspectable."""
from .models.classic import EASE, ExplicitALS, ItemKNN, MostPopular
from .models.fm import FM
from .models.timesvd import TimeSVDPP


def make_model(name: str, parameters: dict):
    classes = {"popular": MostPopular, "knn": ItemKNN, "als": ExplicitALS,
               "ease": EASE, "fm": FM, "timesvd": TimeSVDPP,
               "fm_id": FM, "svdpp": TimeSVDPP}
    if name not in classes:
        raise ValueError(f"Unknown model: {name}")
    params = dict(parameters)
    if name == "fm_id":
        params["metadata"] = False
    if name == "svdpp":
        params["temporal"] = False
    return classes[name](**params)
