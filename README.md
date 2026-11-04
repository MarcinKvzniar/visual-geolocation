# Visual Geolocation

Predicting the region and GPS coordinates of street-level images, using a European subset of
[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)
([paper](https://osv5m.github.io/cvpr-static.pdf)).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu    # CPU only
uv sync --extra gpu    # CUDA 13.0
```

## Data

The images are not included in this repository. To download them into `data/images/` (about 8 GB, an
estimated 3 hours), run this from the repository root:

```bash
uv run python -m visual_geolocation.data.download
```

Run it again to resume an interrupted download. The list of images and their splits is in
`configs/subset.parquet`. Delete that file to rebuild it from the dataset's metadata (about 3 GB).
