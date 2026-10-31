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
| `--no-compile` | skip `torch.compile`: quicker start, slower inference (see "Speed" below) |
| `--no-cuda-graphs` | compile without CUDA graphs, like `configs/deploy.yaml` and our latency experiments |
| `--device cuda:N` | which GPU |

These work in `test_one_inference.py` only:

| option | what it does |
|---|---|
| `--seed N` | seed for the random input and the model's starting noise |
| `--repeat N` | extra timed runs on the same input, each with a new noise seed |
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

Output of `--repeat 10` on this devbox (RTX PRO 6000 Blackwell, all speed-ups on, 2026-10-31):

```
model loaded in ~57 s, GPU memory in use 23.1 GiB
actions shape  : (32, 48)
all finite     : True
inside training range (+-5%): 100.0% of the numbers
time           : 9.13 s (DiT cache skipped 3 of the denoising steps)     first run: compiles + warm-up
repeat 1 (seed 1): 0.276 s, DiT cache skipped 3 steps
repeat 2 (seed 2): 0.163 s, DiT cache skipped 6 steps
...
repeats: median 0.162 s, fastest 0.161 s
OK
```

It prints `OK` and exits with code 0 when the chunk has shape (32, 48) and every number is finite. It exits with 1
otherwise. On random input the actions are meaningless; the test only shows the model runs end to end.

The first run compiles. It takes ~45 s the very first time, and ~9 s once torch's on-disk compile cache has it.

## Speed

The OpenWAM paper (arXiv 2609.07398, section 3.3, Figure 4) reports ~175 ms per inference for this architecture
(dual-system joint self-attention, Wan2.2-TI2V-5B) on an RTX 5090. It gets there with four speed-ups; the scripts
turn all four on by default:

| speed-up | what it does | setting |
|---|---|---|
| prompt-embedding cache | runs the text encoder once per distinct prompt | `deploy.yaml`, always on |
| video-decode skip | never turns the predicted video back into pixels; only the actions are used | `deploy.yaml`, always on |
| DiT velocity cache | when the prediction barely changes between denoising steps, reuses it instead of running the network (at most 3 skips in a row) | `deploy.yaml`, always on |
| `torch.compile` + CUDA graphs | compiles the denoising loop and replays it as recorded GPU kernels | on; `--no-compile`, `--no-cuda-graphs` |

`configs/deploy.yaml` compiles this architecture without CUDA graphs (`optimization.compile.self_attn.torch_mode:
default`). The paper uses them, so `load_model` switches that one setting to `reduce-overhead`. It does this in the
script only; `deploy.yaml` is not edited.

Measured here, median of 10 inferences with different seeds:

| setting | time per inference |
|---|---|
| nothing (no compile, no DiT cache) | 498 ms |
| DiT cache only | 212 ms |
| DiT cache + compile (= `deploy.yaml`, our latency experiments) | 173 ms |
| DiT cache + compile + CUDA graphs (the scripts' default, as in the paper) | 163 ms |

**The time per inference depends on the noise seed.** The DiT cache skips either 6 or 3 of the 10 denoising steps,
depending on the starting noise. For this checkpoint, 15 of 20 seeds got 6 skips. With all speed-ups on, that is
~162 ms for 6 skips and ~275 ms for 3; without compile, ~210 ms and ~360 ms. A test that reuses one seed sees only
one of the two levels: seed 0 is a 3-skip seed.

## Notes

- **Same settings as our evaluation server.** Loading goes through `build_server_from_config` with
  `configs/deploy.yaml`, as sim-policy-eval's `scripts/openwam_chunk_server.py` does. That means 10 denoising steps,
  the DiT velocity cache and the prompt-embedding cache. Only nothing is served: the inference engine is used
  directly.
- **Reproducibility.** With the same input and seed, the actions come out identical. Compiled and uncompiled
  runs differ by float rounding (about 1e-3).
