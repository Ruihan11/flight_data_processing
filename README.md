# ✈️ Flight Data Processing End‑to‑End Pipeline

This repository contains a **bash workflow (`run.sh`)** that converts raw
ADS‑B/flight‑track CSVs into engineered features and merges aircraft‑database
labels for downstream machine‑learning tasks (e.g. deviance detection).


---
## dataset

Please download the [dataset](https://theairlab.org/trajair/)

## ⚡ Quick Start  

```bash
# 1) Install uv (skip if already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# 2) Create virtual‑env & install minimal deps
uv venv
uv pip install argparse pandas matplotlib numpy scipy  # add your ML libs here

chmod +x run.sh
./run.sh -j$(nproc)      # -jN optional: parallel jobs for heavy steps

```

