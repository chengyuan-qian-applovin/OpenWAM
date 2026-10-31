# yam_sharpa: load and test jiangwei's `fingertips_abs` checkpoint

This folder adds two scripts to upstream OpenWAM. They download yjw's (jiangwei-yu) `fingertips_abs`
checkpoint, step 100000, which is the one our inference-latency experiments use, and run it once on random input.

The checkpoint is on GCS at `gs://foundational-research/yjw/action_repr_ablation/2026-10-09_22-09-08`. It takes two
camera pictures, the robot's 48-number hand state and a text instruction. It returns the next 32 actions, 32 steps of
48 numbers each (1.6 s at 20 Hz).

## What changed in this repo

| path | change |
|---|---|
| `scripts/yam_sharpa/download_and_load_checkpoint.py` | new: download from GCS, load onto a GPU |
| `scripts/yam_sharpa/test_one_inference.py` | new: one inference on random input, with checks |
| `scripts/yam_sharpa/README.md` | new: this file |

That is the whole change. OpenWAM's own code is unchanged, and the base is upstream `main` (`3950b6d`).

The repo already ignores the two local folders the setup creates: `.venv/` (the Python environment) and
`assets/checkpoints/` (the downloaded checkpoint).

### Why no changes to OpenWAM are needed

jiangwei trained this checkpoint on upstream commit `48bd67b` plus their own changes. Their changes are not in git; they
share the code as a tarball, `gs://foundational-research/yjw/2610_policy_model/code/openwam_code_devbox.tgz`.

- **Inference code.** Between `48bd67b` and `main`, upstream did not change `openwam/deploy/`, `openwam/model/` or
  `openwam/dataloader/transforms/`. So upstream's loader and inference engine are the ones the checkpoint was
  evaluated with.
- **The one piece we need.** This checkpoint sees one picture: the ego camera on top and the chest camera below, each
  resized to 240 x 320, into a 480 x 320 canvas. The config calls this `multiview_layout: vstack`. Upstream's image
  preprocessor does not know this layout. Rather than patch OpenWAM, `test_one_inference.py` builds the picture
  itself (`stack_cameras`, about 10 lines) with the same resize function. It was checked to be pixel-identical to
  jiangwei's `assemble_vstack_layout`.
- **The rest of jiangwei's changes are not needed here.**
  - Training only: the `yam_sharpa` dataloader, trainer and visualization changes.
  - Other checkpoints only: the relative-action decoder (`*_rel`) and the 80-number action layout (`unify_*`) in
    `deploy/model_loader.py` and `deploy/engine.py`. For `fingertips_abs` both stay off.

  Their full code is kept as a single commit (`e829540`) on the local branch `yjw/yam-sharpa`, so it can be read or
  diffed: `git diff 48bd67b yjw/yam-sharpa`. Nothing here uses it.

## How to run

Run everything from the OpenWAM folder.

**1. Python environment (once, about 3 min).** This uses jiangwei's exact package list from their tarball (torch
2.7.1+cu128, Python 3.10.20), the same list our Modal image uses:

```bash
cd /root/Action-Chunking/OpenWAM
uv python install 3.10.20
uv venv --python 3.10.20 .venv
DS_BUILD_OPS=0 uv pip install --python .venv/bin/python \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match -r /root/data/openwam_code/05_devbox_migration/requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

**2. Download the checkpoint (once, about 25 GB, under a minute here).** This needs `gcloud` with read access to
`gs://foundational-research`. Files already on disk are skipped, so rerunning is safe.

```bash
.venv/bin/python scripts/yam_sharpa/download_and_load_checkpoint.py --download-only
```

It downloads `config.yaml`, `checkpoint_step_100000.safetensors`, `normalization_stats.npy` and `tokenizer/` into
`assets/checkpoints/fingertips_abs_100k/`. It skips `accel_state_step_100000/`, about 110 GB of optimizer state
that only resuming training needs.

**3. Load the model and print what it expects (about 1 min, about 23 GB of GPU memory):**

```bash
.venv/bin/python scripts/yam_sharpa/download_and_load_checkpoint.py --skip-download
```

**4. One inference on random input:**

```bash
.venv/bin/python scripts/yam_sharpa/test_one_inference.py                        # load + 1 inference
.venv/bin/python scripts/yam_sharpa/test_one_inference.py --repeat 2             # + 2 timed repeats
.venv/bin/python scripts/yam_sharpa/test_one_inference.py --save-input in.png    # save the picture the model sees
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/yam_sharpa/test_one_inference.py # pick a GPU
```

### Options

These work in both scripts:

| option | what it does |
|---|---|
| `--ckpt-dir DIR` | use a checkpoint folder elsewhere, e.g. `/root/data/openwam_ckpt/fingertips_abs_100k` |
| `--compile` | `torch.compile` the model, as our experiments did. Each inference is a bit faster, but the first one waits a few minutes. Off by default. |
| `--device cuda:N` | which GPU |

These work in `test_one_inference.py` only:

| option | what it does |
|---|---|
| `--seed N` | seed for the random input and the model's starting noise |
| `--repeat N` | extra timed runs on the same input |
| `--save-input PATH` | save the stacked 480 x 320 picture as a PNG |

## What the test does and what to expect

`test_one_inference.py` runs these steps:

1. Loads the model.
2. Makes random input:
   - two random 480 x 640 RGB pictures, the size our simulator renders;
   - a 48-number state, each number random between its 1st and 99th percentile in the training data. It is in range,
     but it is not a real pose;
   - a prompt.
3. Stacks the two pictures.
4. Runs 10 denoising steps.
5. Checks the 32 x 48 result.

Output on this devbox (RTX PRO 6000 Blackwell, no compile, 2026-10-30):

```
model loaded in 56 s, GPU memory in use 23.1 GiB
actions shape  : (32, 48)
all finite     : True
inside training range (+-5%): 100.0% of the numbers
time           : 1.75 s              (first run: warm-up)
repeat 1: 0.37 s, same actions as the first run: True
OK
```

It prints `OK` and exits with code 0 when the chunk has shape (32, 48) and every number is finite. It exits with 1
otherwise. On random input the actions are meaningless; the test only shows the model runs end to end.

## Notes

- **Same settings as our evaluation server.** Loading goes through `build_server_from_config` with
  `configs/deploy.yaml`, as sim-policy-eval's `scripts/openwam_chunk_server.py` does. That means 10 denoising steps,
  the DiT velocity cache and the prompt-embedding cache. Only nothing is served: the inference engine is used
  directly.
- **Reproducibility.** With the same input and seed, the actions come out identical. Compiled and uncompiled
  runs differ by float rounding (about 1e-3).
