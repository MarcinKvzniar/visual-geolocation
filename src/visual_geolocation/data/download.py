"""Download the subset's images: `uv run python -m visual_geolocation.data.download`.

Run from the repository root. Run it again to resume an interrupted download.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from remotezip import RemoteZip
from tqdm import tqdm

from visual_geolocation.data.subset import select_subset

REPO_ID = "osv5m/osv5m"
REVISION = "cff33609b56b54d8743b7ee7a416eb8433e9a681"
RAW_DIR = Path("data/raw")
IMAGE_DIR = Path("data/images")
SUBSET_PATH = Path("configs/subset.parquet")
TRAIN_SHARDS = [f"images/train/{i:02d}.zip" for i in range(98)]
TEST_SHARDS = [f"images/test/{i:02d}.zip" for i in range(5)]
WORKERS = 16


def load_subset() -> pd.DataFrame:
    """Load the committed subset, or rebuild it from the metadata if the file was deleted."""
    if not SUBSET_PATH.exists():
        metadata = pd.concat([_read_metadata("train.csv"), _read_metadata("test.csv")])
        select_subset(metadata).to_parquet(SUBSET_PATH, index=False)
    return pd.read_parquet(SUBSET_PATH)


def download_images(image_ids: set[str]) -> None:
    """Download the images that are not on disk yet, and fail if any can't be found."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = image_ids - _downloaded_ids()
    if wanted:
        shards = TRAIN_SHARDS + TEST_SHARDS
        with ThreadPoolExecutor(WORKERS) as pool:
            jobs = pool.map(partial(_download_from_shard, wanted=wanted), shards)
            list(tqdm(jobs, total=len(shards), desc="Shards"))
    if missing := image_ids - _downloaded_ids():
        raise RuntimeError(f"{len(missing)} images were not found in the shards")


def _read_metadata(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        REPO_ID, filename, repo_type="dataset", revision=REVISION, local_dir=RAW_DIR
    )
    return pd.read_csv(
        path,
        usecols=["id", "latitude", "longitude", "country", "creator_id"],
        dtype={"id": str, "country": str, "creator_id": str},
        keep_default_na=False,
    )


def _download_from_shard(shard: str, wanted: set[str]) -> None:
    resolve_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{shard}"
    with RemoteZip(requests.head(resolve_url).headers["location"]) as archive:
        for name in archive.namelist():
            image_id = Path(name).stem
            if image_id in wanted:
                partial_file = IMAGE_DIR / f"{image_id}.part"
                partial_file.write_bytes(archive.read(name))
                partial_file.rename(IMAGE_DIR / f"{image_id}.jpg")


def _downloaded_ids() -> set[str]:
    return {path.stem for path in IMAGE_DIR.glob("*.jpg")}


if __name__ == "__main__":
    download_images(set(load_subset()["id"]))
