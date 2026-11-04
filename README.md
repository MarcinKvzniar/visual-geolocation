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

The dataset is not included in this repository. Instructions for recreating the subset will be added
along with the acquisition script.
