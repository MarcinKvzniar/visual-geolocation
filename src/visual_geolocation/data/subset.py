"""Select the subset: contributor-disjoint splits that are at least 1 km apart."""

import numpy as np
import pandas as pd
import splitkit
from sklearn.neighbors import BallTree

COUNTRIES = ["DE", "FR", "GB", "SE", "IT", "PL", "ES", "RO", "PT", "FI", "GR", "IE", "NO", "BY"]
SEED = 0
TRAIN_SIZE, TRAIN_CAP = 10_000, 2_000  # images per country, images per contributor in a country
HELD_OUT_SIZE, HELD_OUT_CAP = 1_000, 200  # same cap for val and for test
MIN_DISTANCE_KM = 1.0
EARTH_RADIUS_KM = 6371.0088


def select_subset(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return the images of the subset, with a `split` column of train, val or test."""
    mask = metadata["country"].isin(COUNTRIES) & (metadata["creator_id"] != "")
    pool = metadata[mask].sort_values("id").reset_index(drop=True)
    result = splitkit.split(
        pool,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        group_col="creator_id",
        label_col="country",
        strategy="annealing",
        seed=SEED,
        max_evals=300_000,
    )
    pool = result.assign_column(pool, "creator_id")
    splits = dict(tuple(pool.groupby("split")))

    rng = np.random.default_rng(SEED)
    test = _sample(splits["test"], HELD_OUT_SIZE, HELD_OUT_CAP, rng)
    val = _sample(_far_from(splits["val"], test), HELD_OUT_SIZE, HELD_OUT_CAP, rng)
    held_out = pd.concat([val, test])
    train = _sample(_far_from(splits["train"], held_out), TRAIN_SIZE, TRAIN_CAP, rng)
    return pd.concat([train, held_out], ignore_index=True)


def _sample(images: pd.DataFrame, size: int, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly pick up to `size` images per country and `cap` per contributor in a country."""
    samples = []
    for country in COUNTRIES:
        candidates = images[images["country"] == country]
        candidates = candidates.iloc[rng.permutation(len(candidates))]
        samples.append(candidates[candidates.groupby("creator_id").cumcount() < cap].head(size))
    return pd.concat(samples)


def _far_from(images: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    """Keep the images at least `MIN_DISTANCE_KM` away from all of `others`."""
    columns = ["latitude", "longitude"]
    tree = BallTree(np.radians(others[columns].to_numpy()), metric="haversine")
    nearby = tree.query_radius(
        np.radians(images[columns].to_numpy()),
        r=MIN_DISTANCE_KM / EARTH_RADIUS_KM,
        count_only=True,
    )
    return images[nearby == 0]
