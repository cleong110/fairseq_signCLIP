#!/usr/bin/env python3
"""
Inspect a PyTorch .pt or .pth checkpoint file and summarize its contents.
Supports recursive inspection (depth controlled).
"""

import argparse
import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def load_checkpoint(path: Path, weights_only: bool) -> Any:
    """
    Load a PyTorch checkpoint file.
    """
    if weights_only:
        import argparse as ap
        torch.serialization.add_safe_globals([ap.Namespace])
        return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_state_dict(state_dict: Mapping[str, Any], max_items: int = 20) -> None:
    """
    Summarize a state_dict: list parameter names and shapes.
    """
    logging.info("state_dict has %d parameters.", len(state_dict))
    for i, (k, v) in enumerate(state_dict.items()):
        if hasattr(v, "shape"):
            shape_str = tuple(v.shape)  # torch.Tensor
        else:
            shape_str = type(v).__name__
        logging.info("  %s: %s", k, shape_str)
        if i + 1 >= max_items:
            logging.info("  ... (%d more parameters)", len(state_dict) - max_items)
            break

def summarize_obj(obj: Any, depth: int, max_depth: int, prefix: str = "") -> None:
    """
    Recursively summarize an object up to max_depth.
    Prints tensor shapes instead of raw content.
    """
    import torch  # ensure torch is available here

    if isinstance(obj, Mapping):
        logging.info("%s(dict with %d keys)", prefix, len(obj))
        if depth < max_depth:
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    logging.info(
                        "%s  Key: %s (Tensor) shape: %s dtype: %s",
                        prefix,
                        k,
                        tuple(v.shape),
                        v.dtype,
                    )
                else:
                    logging.info("%s  Key: %s (type: %s)", prefix, k, type(v).__name__)
                    summarize_obj(v, depth + 1, max_depth, prefix + "    ")
        elif "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            summarize_state_dict(obj["state_dict"])
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        logging.info("%s(list/seq with %d elements)", prefix, len(obj))
        if depth < max_depth:
            for i, item in enumerate(obj[:10]):
                if isinstance(item, torch.Tensor):
                    logging.info(
                        "%s  [%d] Tensor shape: %s dtype: %s",
                        prefix,
                        i,
                        tuple(item.shape),
                        item.dtype,
                    )
                else:
                    logging.info("%s  [%d] type: %s", prefix, i, type(item).__name__)
                    summarize_obj(item, depth + 1, max_depth, prefix + "    ")
            if len(obj) > 10:
                logging.info("%s  ... (%d more elements)", prefix, len(obj) - 10)
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        # Generic tensor-like object (covers torch.Tensor or numpy arrays)
        logging.info("%sTensor-like shape: %s dtype: %s", prefix, tuple(obj.shape), obj.dtype)
    else:
        # Base case: not a mapping, sequence, or tensor-like
        preview = str(obj)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        logging.info("%s%s", prefix, preview)



def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint (.pt or .pth).")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        default=False,
        help="Use safe weights_only loading mode (default: False).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="How deeply to inspect nested dict/list structures (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt_path = args.checkpoint
    if not ckpt_path.is_file():
        logging.error("File not found: %s", ckpt_path)
        return

    logging.info("Inspecting checkpoint: %s", ckpt_path)

    for _ in tqdm(range(1), desc="Loading checkpoint"):
        try:
            obj = load_checkpoint(ckpt_path, weights_only=args.weights_only)
        except (pickle.UnpicklingError, RuntimeError, EOFError) as e:
            logging.error("Failed to load checkpoint: %s", e)
            return

    summarize_obj(obj, depth=0, max_depth=args.depth)


if __name__ == "__main__":
    main()
