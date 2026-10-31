"""Run one inference of the OpenWAM ``fingertips_abs`` checkpoint on random input.

A smoke test: does the model load, accept inputs of the right shape and return an action chunk of the right shape,
with finite numbers, in a sensible time? Random input means the actions themselves are meaningless.

What happens, step by step
--------------------------
1. Load the model (``download_and_load_checkpoint.load_model``, ~1-2 min).
2. Make a random observation, shaped like what our simulator sends:
     - two random 480 x 640 RGB pictures, one per camera (ego_view, chest_view);
     - a random 48-number state, each number drawn between the 1st and 99th percentile of that number in the
       training data (``normalization_stats.npy``), so it is in a range the model has seen. It is NOT a
       physically valid pose (e.g. the wrist axes are not perpendicular), which is fine for a smoke test;
     - a task instruction.
3. Put the two pictures together the way the model was trained: each resized to 240 x 320 and stacked, ego on top,
   into one 480 x 320 picture (``stack_cameras``).
4. Run the model once: starting from random noise it removes the noise in 10 steps, producing the future video
   (internally, not decoded) and the 32 x 48 action chunk together.
5. Print the chunk's shape, whether it is finite and inside the training range, and the time taken.
   With ``--repeat N`` it runs N more times on the same input, to show the steady-state speed.

Usage (from the OpenWAM folder, with its venv)::

    .venv/bin/python scripts/yam_sharpa/test_one_inference.py
    .venv/bin/python scripts/yam_sharpa/test_one_inference.py --repeat 3 --save-input /tmp/model_input.png
    .venv/bin/python scripts/yam_sharpa/test_one_inference.py --ckpt-dir /root/data/openwam_ckpt/fingertips_abs_100k

Run ``download_and_load_checkpoint.py --download-only`` first (or point ``--ckpt-dir`` at a copy).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # download_and_load_checkpoint (same folder)

from download_and_load_checkpoint import DEFAULT_CKPT_DIR, add_speed_arguments, describe, load_model  # noqa: E402

CAMERAS = ("ego_view", "chest_view")  # config.yaml dataloader.camera_layout, top to bottom
CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640  # what our MuJoCo simulator renders (any size works: they get resized)
PROMPT = "pick up the apple and place it in the bowl"


def random_observation(ckpt_dir: Path, seed: int) -> dict:
    """A random observation: two camera pictures, a 48-number state and a prompt.

    Args:
        ckpt_dir: the checkpoint folder (its normalization_stats.npy sets the range of the random state).
        seed: random seed, so a rerun gets the same input.

    Returns:
        ``{"images": {camera name: PIL image}, "state": (48,) float32, "prompt": str}``.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)

    # Two pictures of random coloured pixels (values 0-255, 3 colour channels).
    images = {
        cam: Image.fromarray(rng.integers(0, 256, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)) for cam in CAMERAS
    }

    # The state: 48 numbers, each uniformly random between its 1st and 99th percentile in the training data.
    # normalization_stats.npy is a dict; "fingertips_abs_state" holds the state's per-number statistics.
    stats = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()["fingertips_abs_state"]
    state = rng.uniform(np.asarray(stats["q01"]), np.asarray(stats["q99"])).astype(np.float32)

    return {"images": images, "state": state, "prompt": PROMPT}


def stack_cameras(images: dict, height: int, width: int):
    """Put the camera pictures into the one picture the model takes: stacked top to bottom, in ``CAMERAS`` order.

    This is the "vstack" layout of the checkpoint's config (``dataloader.multiview_layout``). It is the only piece
    of jiangwei's code changes that inference with this checkpoint needs, so it is reproduced here rather than
    patched into OpenWAM: it does what their ``assemble_vstack_layout`` (training and their deploy server) does,
    using the same resize function, so the picture matches training pixel for pixel.

    Each camera gets an equal-height strip: with two cameras on a 480 x 320 canvas, each 480 x 640 picture is
    shrunk to 240 x 320 (it keeps its 4:3 shape) and the two strips are pasted one above the other.

    Args:
        images: ``{camera name: PIL image}`` with every camera of ``CAMERAS``, any size.
        height, width: the canvas size, from the config (``inference.height`` / ``inference.width``: 480, 320).

    Returns:
        PIL image of ``height`` x ``width``.
    """
    from PIL import Image

    from openwam.dataloader.transforms.multiview import _stretch_resize  # a plain bilinear resize

    strip_height = height // len(CAMERAS)
    canvas = Image.new("RGB", (width, height))
    for i, cam in enumerate(CAMERAS):
        canvas.paste(_stretch_resize(images[cam].convert("RGB"), strip_height, width), (0, i * strip_height))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--device", default="cuda")
    add_speed_arguments(ap)  # --no-compile, --no-cuda-graphs
    ap.add_argument("--seed", type=int, default=0, help="seed for the random input and for the model's noise")
    ap.add_argument(
        "--repeat", type=int, default=0, help="extra timed runs on the same input, run i with noise seed --seed + i"
    )
    ap.add_argument("--save-input", type=Path, default=None, help="save the stacked picture the model sees (PNG)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # --- 1. Load the model --------------------------------------------------------------------------------------
    model = load_model(
        args.ckpt_dir, device=args.device, compile=not args.no_compile, cuda_graphs=not args.no_cuda_graphs
    )
    describe(model)
    # Check the checkpoint really uses the layout stack_cameras builds (it would be silently wrong otherwise).
    assert list(model.cfg.dataloader.camera_layout) == [f"observation.images.{c}" for c in CAMERAS]
    assert model.cfg.dataloader.multiview_layout == "vstack"

    # --- 2. A random observation --------------------------------------------------------------------------------
    obs = random_observation(args.ckpt_dir, args.seed)
    print("\nRandom input")
    for cam, img in obs["images"].items():
        print(f"  {cam}: {img.height} x {img.width} RGB")
    print(f"  state: {obs['state'].shape[0]} numbers, first 3 = {np.round(obs['state'][:3], 3)}")
    print(f"  prompt: {obs['prompt']!r}")

    # --- 3. The one picture the model sees ----------------------------------------------------------------------
    picture = stack_cameras(obs["images"], int(model.cfg.inference.height), int(model.cfg.inference.width))
    print(f"  -> the model sees one {picture.height} x {picture.width} picture (ego on top, chest below)")
    if args.save_input:
        picture.save(args.save_input)
        print(f"     saved to {args.save_input}")

    # The engine's input ("conditions"), the same keys OpenWAM's WAMPolicy._build_conditions builds:
    #   first_frame_image  the current picture; the model predicts the video that follows it
    #   prompt             the instruction
    #   proprio            the raw 48-number state; the engine scales it to [-1, 1] itself
    #   seed               seeds the starting noise: the same input and seed give the same actions (set in run_once)
    conditions = {"first_frame_image": [picture], "prompt": obs["prompt"], "proprio": obs["state"]}

    # --- 4. Run the model ---------------------------------------------------------------------------------------
    import torch

    def run_once(seed: int) -> tuple[np.ndarray, float, int]:
        """One inference with noise seed ``seed``. Returns (actions, seconds, denoising steps the DiT cache skipped)."""
        torch.cuda.synchronize()  # GPU work runs asynchronously; sync so the timer measures the whole inference
        t0 = time.monotonic()
        with torch.inference_mode():
            actions = model.engine.generate({**conditions, "seed": seed})["actions"]
        torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        if hasattr(actions, "detach"):  # a torch tensor -> numpy
            actions = actions.detach().float().cpu().numpy()
        # The DiT cache counts the steps it skipped in the inference that just ran; most of the time per inference is
        # the steps it did not skip, so this number explains most of the variation in time.
        cache = model.engine._dit_cache
        skipped = cache.stats["total_skips"] if cache is not None else 0
        return np.asarray(actions, dtype=np.float32), seconds, skipped

    print("\nRunning one inference (the first one also compiles, warms up the GPU kernels and caches the prompt) ...")
    actions, seconds, skipped = run_once(args.seed)

    # --- 5. Check the result ------------------------------------------------------------------------------------
    # The actions come back already un-scaled, i.e. in the same units as the state (metres, unit vectors).
    stats = np.load(args.ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()["fingertips_abs"]
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    margin = 0.05 * (hi - lo)  # a little slack: the model may go slightly past the training min / max
    inside = (actions >= lo - margin) & (actions <= hi + margin)
    print("\nResult")
    print(f"  actions shape  : {actions.shape}  (steps x numbers; expected (32, 48))")
    print(f"  all finite     : {bool(np.isfinite(actions).all())}")
    print(f"  inside training range (+-5%): {inside.mean():.1%} of the numbers")
    print(f"  first step, left wrist position [m]: {np.round(actions[0, :3], 3)}")
    print(f"  time           : {seconds:.2f} s (DiT cache skipped {skipped} of the denoising steps)")

    # Repeats: same pictures, state and prompt, a different starting noise each time (as successive queries of a
    # rollout get). The noise decides how many steps the DiT cache can skip (3 or 6 of 10 here), which is why the
    # time jumps between two levels, e.g. ~0.16 s and ~0.27 s with compile on.
    times = []
    for i in range(1, args.repeat + 1):
        _, seconds, skipped = run_once(args.seed + i)
        times.append(seconds)
        print(f"  repeat {i} (seed {args.seed + i}): {seconds:.3f} s, DiT cache skipped {skipped} steps")
    if times:
        print(f"  repeats: median {np.median(times):.3f} s, fastest {min(times):.3f} s")

    ok = actions.shape == (32, 48) and np.isfinite(actions).all()
    print("\nOK" if ok else "\nFAILED: unexpected shape or non-finite actions")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
