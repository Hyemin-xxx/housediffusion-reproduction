"""
HouseDiffusion sampling script — Apple Silicon (M1/M2/M3) compatible.
v2: object array 처리 강화 + 무한루프 방지
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch as th

from house_diffusion import logger
from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)

th.set_default_dtype(th.float32)


def pick_device() -> th.device:
    if th.backends.mps.is_available() and th.backends.mps.is_built():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def device_sync(device: th.device) -> None:
    if device.type == "cuda":
        th.cuda.synchronize()
    elif device.type == "mps":
        th.mps.synchronize()


def device_empty_cache(device: th.device) -> None:
    if device.type == "cuda":
        th.cuda.empty_cache()
    elif device.type == "mps" and hasattr(th.mps, "empty_cache"):
        th.mps.empty_cache()


def _np_to_tensor(arr: np.ndarray, device: th.device) -> th.Tensor:
    if arr.dtype == np.bool_:
        t = th.from_numpy(arr.copy())
    elif np.issubdtype(arr.dtype, np.integer):
        t = th.from_numpy(arr.astype(np.int64, copy=False))
    elif np.issubdtype(arr.dtype, np.floating):
        t = th.from_numpy(arr.astype(np.float32, copy=False))
    else:
        t = th.from_numpy(np.asarray(arr))
    return t.to(device)


def _resolve_object_array(arr: np.ndarray) -> np.ndarray:
    """object dtype numpy를 numeric으로 풀어낸다. 핵심 함수."""
    if arr.dtype != object:
        return arr

    flat = arr.flatten()
    if flat.size == 0:
        return arr.astype(np.float32)

    items = [np.asarray(x) for x in flat]
    first_shape = items[0].shape
    same_shape = all(it.shape == first_shape for it in items)

    if same_shape:
        stacked = np.stack(items, axis=0)
        new_shape = arr.shape + first_shape
        return stacked.reshape(new_shape).astype(np.float32, copy=False)

    try:
        ndim = items[0].ndim
        if all(it.ndim == ndim for it in items):
            max_shape = [max(it.shape[d] for it in items) for d in range(ndim)]
            padded = np.zeros((len(items), *max_shape), dtype=np.float32)
            for i, it in enumerate(items):
                slc = tuple(slice(0, s) for s in it.shape)
                padded[(i,) + slc] = it
            new_shape = arr.shape + tuple(max_shape)
            return padded.reshape(new_shape)
    except Exception:
        pass

    raise RuntimeError(
        f"object array를 numeric으로 풀 수 없음: "
        f"outer shape={arr.shape}, inner shape={first_shape}"
    )


def to_device_recursive(obj, device: th.device):
    if isinstance(obj, th.Tensor):
        t = obj.to(device)
        if t.is_floating_point() and t.dtype == th.float64:
            t = t.to(th.float32)
        return t

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = _resolve_object_array(obj)
        return _np_to_tensor(obj, device)

    if isinstance(obj, dict):
        return {k: to_device_recursive(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device_recursive(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device_recursive(v, device) for v in obj)

    return obj


def save_sample(sample: th.Tensor, model_kwargs: dict, idx: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_kwargs = {}
    for k, v in model_kwargs.items():
        if isinstance(v, th.Tensor):
            cpu_kwargs[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            cpu_kwargs[k] = v

    np.savez(
        out_dir / f"sample_{idx:05d}.npz",
        sample=sample.detach().cpu().numpy(),
        **cpu_kwargs,
    )


def main() -> None:
    args = create_argparser().parse_args()
    update_arg_parser(args)

    if getattr(args, "use_fp16", False):
        logger.log("[M1] use_fp16 강제 비활성")
        args.use_fp16 = False

    device = pick_device()
    logger.configure(dir=args.save_dir if hasattr(args, "save_dir") and args.save_dir else None)
    logger.log(f"[M1] using device: {device}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.log("creating data loader...")
    data = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=args.analog_bit,
        target_set=args.target_set,
        set_name=args.set_name,
    )

    logger.log("sampling...")
    out_dir = Path("outputs") / f"set{args.target_set}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    saved = 0
    sample_idx = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    first_error_logged = False

    for batch_idx, items in enumerate(data):
        if saved >= args.num_samples:
            break

        try:
            if isinstance(items, (list, tuple)) and len(items) > 0 and isinstance(items[0], tuple):
                batch = np.stack([it[0] for it in items], axis=0)
                cond_keys = items[0][1].keys()
                model_kwargs = {
                    k: np.stack([it[1][k] for it in items], axis=0)
                    for k in cond_keys
                }
            else:
                batch, model_kwargs = items
        except Exception as e:
            logger.log(f"[M1] batch {batch_idx} 풀기 실패: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.log("[M1] 연속 실패 너무 많음, 중단")
                break
            continue

        try:
            model_kwargs = to_device_recursive(model_kwargs, device)
        except Exception as e:
            consecutive_failures += 1
            if not first_error_logged:
                logger.log(f"[M1] batch {batch_idx} 변환 실패. 첫 에러 traceback:")
                logger.log(traceback.format_exc())
                first_error_logged = True
            else:
                logger.log(f"[M1] batch {batch_idx} 또 실패: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.log(f"[M1] 연속 {consecutive_failures}회 실패 → 중단")
                break
            continue

        consecutive_failures = 0

        batch_shape = batch.shape if isinstance(batch, np.ndarray) else tuple(batch.shape)

        try:
            with th.no_grad():
                sample = sample_fn(
                    model,
                    batch_shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=device,
                )
        except TypeError:
            with th.no_grad():
                sample = sample_fn(
                    model,
                    batch_shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

        save_sample(sample, model_kwargs, sample_idx, out_dir)
        sample_idx += 1
        saved += sample.shape[0]
        device_sync(device)
        device_empty_cache(device)
        logger.log(f"  saved {saved}/{args.num_samples}")

    if saved == 0:
        logger.log("[M1] 샘플이 하나도 안 나왔습니다. 위 traceback 확인.")
    else:
        logger.log(f"done. outputs at {out_dir.resolve()}")


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        dataset="rplan",
        clip_denoised=False,
        num_samples=64,
        batch_size=1,
        use_ddim=False,
        model_path="ckpts/exp/model250000.pt",
        draw_graph=False,
        save_svg=False,
        set_name="eval",
        target_set=8,
        save_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
