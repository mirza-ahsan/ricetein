"""The time-neighbour control: known answers, and no use of validation or test labels."""

import numpy as np
import pandas as pd

from src import shortcut_baselines as sb

CLASSES = ["healthy", "leaf_blast", "brown_spot", "hispa"]


def table():
    return pd.DataFrame({
        "photo_id": list("abcdefgh"),
        "class": ["healthy", "hispa", "hispa", "leaf_blast", "healthy", "brown_spot", "hispa", "hispa"],
        "camera": ["x", "x", "x", "x", "y", "x", "x", "x"],
        "seconds": [0.0, 10.0, 12.0, 100.0, 11.0, 96.0, np.nan, 900.0],
        "round_1": ["train", "test", "train", "train", "test", "val", "test", "train"],
    })


def test_nearest_training_photo_of_the_same_camera_decides():
    out = sb.time_neighbour_round(table(), 1, CLASSES).set_index("photo_id")
    assert out.loc["b", "pred_class"] == "hispa"        # c (2 s away) beats a (10 s away)
    assert out.loc["f", "pred_class"] == "leaf_blast"   # d, 4 s away
    assert out.loc["e", "pred_class"] == "hispa"        # camera y has no training photo: training majority
    assert out.loc["g", "pred_class"] == "hispa"        # no capture time: training majority
    assert out.loc["f", "partition"] == "val" and out.loc["b", "partition"] == "test"
    assert (out[[f"prob_{c}" for c in CLASSES]].sum(axis=1) == 1).all()


def test_labels_of_scored_photos_are_never_used():
    changed = table()
    changed.loc[changed["round_1"] != "train", "class"] = "brown_spot"
    a = sb.time_neighbour_round(table(), 1, CLASSES)["pred_class"].tolist()
    assert a == sb.time_neighbour_round(changed, 1, CLASSES)["pred_class"].tolist()


def test_paper_histogram_ignores_the_leaf_and_its_surroundings(tmp_path):
    from PIL import Image

    image = np.full((400, 400, 3), 235, np.uint8)       # pale paper
    image[150:250, :] = (40, 140, 40)                   # a green leaf across the photo
    path = tmp_path / "leaf.jpg"
    Image.fromarray(image).save(path, quality=100)
    hist = sb.paper_histogram(str(path)).reshape(8, 8, 8)
    assert np.isclose(hist.sum(), 1)
    assert hist[:, 0, 7].sum() > 0.99                   # only colourless, bright pixels were counted
