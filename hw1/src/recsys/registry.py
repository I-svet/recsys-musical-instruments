"""Explicit small registry keeps all model implementations inspectable."""
from models.itemknn import ItemKNN
from models.als import ExplicitALS
from models.ease import EASE
from models.fm import FM
from models.timesvd import TimeSVDPP
from models.popular import MostPopular


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
