"""Download the OpenWAM ``fingertips_abs`` checkpoint and load it onto a GPU.

What this checkpoint is
-----------------------
jiangwei-yu's (yjw) OpenWAM model trained on the YAM arm + Sharpa hand data (``dataloader.type: yam_sharpa``) with
the ``fingertips_abs`` action representation, step 100000. It is the checkpoint our inference-latency experiments
use (``openwam_latency/``). In one sentence: given two camera images, the robot's current hand poses and a text
instruction, it predicts the next 32 robot actions (an "action chunk", 32 steps at 20 Hz = 1.6 s).

    input   ego_view image  (head camera)              any size, e.g. 480 x 640 RGB
            chest_view image (chest camera)            any size, e.g. 480 x 640 RGB
            state, 48 numbers                          per hand (left, right): wrist position + orientation in the
                                                       ego camera frame, and the five fingertip positions
            prompt                                     the task instruction, e.g. "pick up the apple"
    output  actions, 32 x 48 numbers                   32 future targets, same 48-number layout as the state

Where it lives
--------------
    gs://foundational-research/yjw/action_repr_ablation/2026-10-09_22-09-08/

We only download what inference needs (~25 GB). The folder also holds ``accel_state_step_100000/`` (~110 GB, the
optimizer state for resuming training), which we skip.

    config.yaml                          the training config: model shape, cameras, action representation
    checkpoint_step_100000.safetensors   all the weights (video model, VAE, text encoder, action model) in one file
    normalization_stats.npy              min / max of every state and action number in the training data; the
                                         model works on values scaled to [-1, 1] with these
    tokenizer/                           turns the prompt text into tokens for the text encoder

Code it needs
-------------
Upstream OpenWAM as it is: no changes to the library. jiangwei trained this checkpoint on upstream ``48bd67b`` plus
their own ``yam_sharpa`` additions, and the inference code (``openwam/deploy``, ``openwam/model``) has not changed
upstream since then. Of their additions, inference with this checkpoint needs only one: how the two camera images are
put together into one picture ("vstack": ego on top, chest below). Upstream's preprocessor does not know that
layout, so ``test_one_inference.py`` builds the picture itself (``stack_cameras``) instead of using it. Their other
changes are for training, or for the ``*_rel`` / ``unify_*`` checkpoints, which this one is not.

Usage (from the OpenWAM folder, with its venv)::

    .venv/bin/python scripts/yam_sharpa/download_and_load_checkpoint.py                 # download + load
    .venv/bin/python scripts/yam_sharpa/download_and_load_checkpoint.py --download-only
    .venv/bin/python scripts/yam_sharpa/download_and_load_checkpoint.py --ckpt-dir /root/data/openwam_ckpt/fingertips_abs_100k

Downloading needs ``gcloud`` logged in with read access to ``gs://foundational-research``. Files already on disk
are not downloaded again, so rerunning is cheap.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The OpenWAM folder (two levels up from this file), so ``import openwam`` works however the script is started.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

GCS_RUN_DIR = "gs://foundational-research/yjw/action_repr_ablation/2026-10-09_22-09-08"
CKPT_NAME = "checkpoint_step_100000.safetensors"
# The pieces inference needs: (name in the GCS folder, is it a folder?)
NEEDED = [("config.yaml", False), ("normalization_stats.npy", False), ("tokenizer", True), (CKPT_NAME, False)]
# assets/ is git-ignored in this repo, so the 25 GB never ends up in git.
DEFAULT_CKPT_DIR = REPO_ROOT / "assets" / "checkpoints" / "fingertips_abs_100k"

logger = logging.getLogger("yam_sharpa")


def download_checkpoint(ckpt_dir: Path = DEFAULT_CKPT_DIR) -> Path:
    """Copy the files inference needs from GCS into ``ckpt_dir``, skipping the ones already there.

    Args:
        ckpt_dir: local folder to download into (created if missing).

    Returns:
        ``ckpt_dir``, now holding config.yaml, the .safetensors, normalization_stats.npy and tokenizer/.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, is_folder in NEEDED:
        local = ckpt_dir / name
        # A folder counts as downloaded when it has at least one file in it. gcloud writes into a temporary
        # "*_.gstmp" file and renames it at the end, so a half-finished download never looks complete.
        done = any(p.is_file() for p in local.rglob("*")) if is_folder else local.is_file()
        if done:
            logger.info("already here, skipping: %s", local)
            continue
        logger.info("downloading %s/%s -> %s", GCS_RUN_DIR, name, local)
        cmd = ["gcloud", "storage", "cp"] + (["--recursive"] if is_folder else []) + [f"{GCS_RUN_DIR}/{name}"]
        # For a folder, copy into the parent: gcloud then creates ckpt_dir/tokenizer/ itself.
        cmd.append(str(ckpt_dir) if is_folder else str(local))
        t0 = time.monotonic()
        subprocess.run(cmd, check=True)
        logger.info("  done in %.0f s", time.monotonic() - t0)
    return ckpt_dir


@dataclass
class LoadedModel:
    """Everything needed to run inference, returned by :func:`load_model`.

    Attributes:
        engine: OpenWAM's ``JointInferenceEngine``. ``engine.generate(conditions)`` runs one denoising pass and
            returns ``{"actions": (32, 48) array, "video": ...}``.
        cfg: the training config (``config.yaml``) with the deploy settings (``configs/deploy.yaml``) merged on
            top: denoising steps, compile, caches.
    """

    engine: object
    cfg: object


def load_model(
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    device: str = "cuda",
    compile: bool = True,
    cuda_graphs: bool = True,
    denoise_steps: int | None = None,
) -> LoadedModel:
    """Load the checkpoint the same way our evaluation server does (``scripts/openwam_chunk_server.py`` in
    sim-policy-eval): OpenWAM's own deploy code, ``configs/deploy.yaml`` defaults.

    Takes ~1-2 minutes and ~25 GB of GPU memory (the weights are kept in bfloat16).

    Speed-ups. The OpenWAM paper (arXiv 2609.07398, section 3.3) uses four, all on here by default:
      - prompt-embedding cache: the text encoder runs once per distinct prompt, not on every inference
        (deploy.yaml ``optimization.prompt_embed_cache``, always on here);
      - video-decode skip: the predicted video is never turned back into pixels, only the actions are used
        (``optimization.decode_video: false``, always on here);
      - DiT velocity cache: when the model's prediction barely changes from one denoising step to the next, the next
        step reuses it instead of running the network (``optimization.dit_cache``, always on here). How many of the
        10 steps it skips depends on the starting noise: for this checkpoint either 6 (~165 ms per inference
        with compile) or 3 (~275 ms), so the time varies from one inference to the next;
      - compilation (``compile``) with CUDA graphs (``cuda_graphs``), see below.

    Args:
        ckpt_dir: folder with config.yaml, the .safetensors, normalization_stats.npy and tokenizer/.
        device: GPU to load on, e.g. ``"cuda"`` or ``"cuda:1"``.
        compile: ``torch.compile`` the denoising loop. On by default, as in the paper and our experiments: each
            inference is faster afterwards (6-skip case ~210 ms -> ~165 ms), but the first one waits for compilation
            (~45 s; less when torch's on-disk compile cache already has it). The actions differ from uncompiled ones
            only by float rounding (~1e-3).
        cuda_graphs: with ``compile``, also record the compiled loop as CUDA graphs (torch's "reduce-overhead"
            mode), which the paper does: the GPU replays the recorded kernels instead of the CPU launching each one.
            deploy.yaml leaves this off for this architecture (``optimization.compile.self_attn.torch_mode:
            default``); turning it on saves another ~10 ms. Our latency experiments ran with it off.
        denoise_steps: how many denoising steps per inference; None keeps deploy.yaml's 10.

    Returns:
        :class:`LoadedModel`.
    """
    from omegaconf import OmegaConf

    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    # 1. Deploy settings: start from the repo's configs/deploy.yaml and apply our switches.
    deploy_cfg = _load_deploy_yaml(str(REPO_ROOT / "configs" / "deploy.yaml"))
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", bool(compile), merge=False)
    if cuda_graphs:
        # "self_attn" is the compile section of this checkpoint's architecture (dual_system, joint_self_attn).
        OmegaConf.update(deploy_cfg, "optimization.compile.self_attn.torch_mode", "reduce-overhead", merge=False)
    if denoise_steps is not None:
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", int(denoise_steps), merge=False)

    # 2. Build the model. build_server_from_config reads config.yaml, builds the network, loads the weights from
    #    the .safetensors, moves it to the GPU in bfloat16, attaches the normalization stats and wraps it in an
    #    inference engine. It also makes a PolicyServer object (OpenWAM's WebSocket server), but nothing is
    #    started or listening: we only take the engine out of it.
    t0 = time.monotonic()
    server = build_server_from_config(cfg=deploy_cfg, ckpt_dir=str(ckpt_dir), device=device, ckpt_name=CKPT_NAME)
    logger.info("model loaded in %.0f s", time.monotonic() - t0)
    return LoadedModel(engine=server.engine, cfg=server.cfg)


def describe(model: LoadedModel) -> None:
    """Print what the loaded model expects and returns, read from its config."""
    from omegaconf import OmegaConf

    def get(key):
        return OmegaConf.select(model.cfg, key)

    print("\nLoaded OpenWAM checkpoint")
    print(f"  action representation : {get('dataloader.action_mode')}")
    print(
        f"  cameras (top to bottom): {list(get('dataloader.camera_layout'))}, "
        f"layout {get('dataloader.multiview_layout')}"
    )
    print(f"  model input picture   : {get('inference.height')} x {get('inference.width')} pixels (height x width)")
    print(f"  state numbers         : {get('model.architecture.state_dim')}")
    print(
        f"  action chunk          : {int(get('inference.num_frames')) - 1} steps x "
        f"{get('model.architecture.action_dim')} numbers"
    )
    print(f"  denoising steps       : {get('inference.denoise_steps')}")
    print(
        f"  speed-ups             : compile {get('optimization.compile.enabled')} "
        f"(mode {get('optimization.compile.self_attn.torch_mode')}; reduce-overhead = CUDA graphs), "
        f"DiT cache {get('optimization.dit_cache.enabled')}, "
        f"prompt cache {get('optimization.prompt_embed_cache.enabled')}, "
        f"video decode {get('optimization.decode_video')}"
    )
    try:
        import torch

        print(f"  GPU memory in use     : {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001  (just a nice-to-have line)
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help=f"where the checkpoint is / goes (default {DEFAULT_CKPT_DIR})",
    )
    ap.add_argument("--download-only", action="store_true", help="download, do not load")
    ap.add_argument("--skip-download", action="store_true", help="the files are already in --ckpt-dir")
    ap.add_argument("--device", default="cuda")
    add_speed_arguments(ap)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not args.skip_download:
        download_checkpoint(args.ckpt_dir)
    if args.download_only:
        return
    describe(
        load_model(args.ckpt_dir, device=args.device, compile=not args.no_compile, cuda_graphs=not args.no_cuda_graphs)
    )


def add_speed_arguments(ap: argparse.ArgumentParser) -> None:
    """The ``--no-compile`` / ``--no-cuda-graphs`` switches, shared with test_one_inference.py."""
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile: quicker start, slower inference")
    ap.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="compile without CUDA graphs, as configs/deploy.yaml and our latency experiments",
    )


if __name__ == "__main__":
    main()
