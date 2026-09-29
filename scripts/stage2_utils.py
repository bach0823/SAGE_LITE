"""
SAGE-Lite Stage-2 Utilities & Checkpoint/RNG State Management.

Dedicated helpers for Stage-2 initialization and checkpoint/RNG handling:
- load_stage1_checkpoint_for_stage2: loads best Stage-1 weights, resolves baseline metrics & Stage-1 epochs
- resolve_stage2_rng_checkpoint: resolves path to Stage-1 RNG checkpoint (explicit or auto-detected)
- load_stage2_rng_checkpoint: loads RNG checkpoint dictionary safely
- restore_rng_states: restores 5 RNG states (torch CPU, torch CUDA, numpy, python, dataloader generator)
- restore_scaler_state: restores GradScaler state dict
- validate_checkpoint_compatibility: validates model_type and p3_mode compatibility
"""

import os
import sys
import json
import random
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def validate_checkpoint_compatibility(
    ckpt_data: Dict[str, Any],
    model_type: str = "B2",
    p3_mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Validate model type and P3 mode compatibility between checkpoint and config."""
    ckpt_type = ckpt_data.get('model_type')
    if ckpt_type and ckpt_type != model_type:
        if logger:
            logger.warning(f"Checkpoint model_type '{ckpt_type}' differs from config '{model_type}'")
    ckpt_p3 = ckpt_data.get('p3_mode')
    if ckpt_p3 is not None and p3_mode is not None and ckpt_p3 != p3_mode:
        if logger:
            logger.warning(f"Checkpoint p3_mode '{ckpt_p3}' differs from config '{p3_mode}'")


def load_stage1_checkpoint_for_stage2(
    args: Any,
    config: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
    output_dir: str,
    model_type: str = "B2",
    p3_mode: Optional[str] = None,
    stage1_max: int = 20,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], float, float, int]:
    """
    Load Stage 1 checkpoint for Stage 2 start (--stage2-only):
    1. Loads model weights from generic_ckpt_path or default output_dir/best_model_*_stage1.pth
    2. Resolves global baseline metrics (best_dice, best_loss) with adjacent checkpoint fallback & CLI overrides
    3. Resolves actual epochs used by Stage 1 (--stage1-epochs-used > stage1_completion.json > ckpt epoch)
    Returns: (stage1_data, global_best_dice, global_best_loss, epochs_used_so_far)
    """
    stage1_ckpt_path = (
        getattr(args, 'checkpoint', None)
        or config.get('checkpoint')
        or os.path.join(output_dir, f"best_model_{model_type.lower()}_stage1.pth")
    )
    if not os.path.exists(stage1_ckpt_path):
        if logger:
            logger.error(f"Cannot find Stage 1 checkpoint at {stage1_ckpt_path}")
        sys.exit(1)

    if logger:
        logger.info(f"Loading Stage 1 checkpoint for --stage2-only: {stage1_ckpt_path}")
    stage1_data = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(stage1_data, model_type, p3_mode, logger)
    model.load_state_dict(stage1_data.get('model_state_dict', stage1_data))

    global_best_dice = float(stage1_data.get('best_dice', 0.0))
    global_best_loss = float(stage1_data.get('best_loss', float('inf')))

    # Baseline resolution: check adjacent checkpoint if best_dice is not recorded in stage1_data
    if global_best_dice == 0.0 and getattr(args, 'best_dice', None) is None:
        dirname = os.path.dirname(stage1_ckpt_path)
        candidate_best = os.path.join(dirname, f"best_model_{model_type.lower()}_stage1.pth")
        if os.path.exists(candidate_best) and candidate_best != stage1_ckpt_path:
            try:
                best_meta = torch.load(candidate_best, map_location='cpu', weights_only=False)
                global_best_dice = float(best_meta.get('best_dice', 0.0))
                global_best_loss = float(best_meta.get('best_loss', float('inf')))
                if logger:
                    logger.info(f"Loaded baseline best metrics from adjacent checkpoint {candidate_best}: Dice={global_best_dice:.4f}, Loss={global_best_loss:.4f}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not load metadata from {candidate_best}: {e}")

    # Explicit CLI overrides take top precedence
    if getattr(args, 'best_dice', None) is not None:
        global_best_dice = float(args.best_dice)
    if getattr(args, 'best_loss', None) is not None:
        global_best_loss = float(args.best_loss)

    # Determine epochs used by Stage 1
    if getattr(args, 'stage1_epochs_used', None) is not None:
        epochs_used_so_far = int(args.stage1_epochs_used)
        if logger:
            logger.info(f"Using explicitly provided --stage1-epochs-used: {epochs_used_so_far}")
    else:
        completion_file = os.path.join(os.path.dirname(stage1_ckpt_path), "stage1_completion.json")
        if not os.path.exists(completion_file):
            completion_file = os.path.join(output_dir, "stage1_completion.json")
        if os.path.exists(completion_file):
            try:
                with open(completion_file, 'r') as f:
                    meta = json.load(f)
                    epochs_used_so_far = int(meta.get('epochs_used', stage1_max))
                if logger:
                    logger.info(f"Loaded actual Stage 1 epochs from {completion_file}: {epochs_used_so_far}")
            except Exception as e:
                epochs_used_so_far = int(stage1_data.get('epoch', stage1_max))
                if logger:
                    logger.warning(f"Could not read {completion_file} ({e}), falling back to checkpoint epoch: {epochs_used_so_far}")
        else:
            epochs_used_so_far = int(stage1_data.get('epoch', stage1_max))
            if logger:
                logger.info(f"No stage1 completion file found; using Stage 1 checkpoint epoch: {epochs_used_so_far}")

    if logger:
        logger.info(
            f"Stage 2 starting from Stage 1: Global Best Dice={global_best_dice:.4f}, "
            f"Global Best Loss={global_best_loss:.4f}, Stage 1 Epochs Used={epochs_used_so_far}"
        )

    return stage1_data, global_best_dice, global_best_loss, epochs_used_so_far


def resolve_stage2_rng_checkpoint(
    explicit_path: Optional[str] = None,
    model_type: str = "B2",
    generic_ckpt_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Resolve RNG checkpoint location with precedence:
    1. Explicit --rng-checkpoint path
    2. Adjacent last_model_*_stage1.pth in same folder as generic_ckpt_path
    3. last_model_*_stage1.pth in output_dir
    """
    if explicit_path is not None:
        if os.path.exists(explicit_path):
            return explicit_path
        if logger:
            logger.error(f"Cannot find specified RNG checkpoint: {explicit_path}")
        sys.exit(1)

    candidates = []
    if generic_ckpt_path:
        ckpt_dir = os.path.dirname(generic_ckpt_path)
        candidates.append(os.path.join(ckpt_dir, f"last_model_{model_type.lower()}_stage1.pth"))
    if output_dir:
        candidates.append(os.path.join(output_dir, f"last_model_{model_type.lower()}_stage1.pth"))

    for cand in candidates:
        if os.path.exists(cand):
            if logger:
                logger.info(f"Auto-detected Stage 1 RNG checkpoint: {cand}")
            return cand

    return None


def load_stage2_rng_checkpoint(
    explicit_path: Optional[str] = None,
    model_type: str = "B2",
    generic_ckpt_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve and load the RNG checkpoint dictionary."""
    rng_ckpt_path = resolve_stage2_rng_checkpoint(
        explicit_path=explicit_path,
        model_type=model_type,
        generic_ckpt_path=generic_ckpt_path,
        output_dir=output_dir,
        logger=logger,
    )
    if rng_ckpt_path and os.path.exists(rng_ckpt_path):
        if logger:
            logger.info(f"Restoring Stage 1 RNG and Scaler state from: {rng_ckpt_path}")
        return torch.load(rng_ckpt_path, map_location='cpu', weights_only=False)
    return None


def restore_rng_states(
    rng_data: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Restore all 5 RNG states from checkpoint dictionary:
    1. torch CPU RNG state
    2. torch CUDA RNG state (if available)
    3. NumPy RNG state
    4. Python standard random state
    5. DataLoader torch.Generator state
    """
    if 'rng_state' in rng_data and rng_data['rng_state'] is not None:
        torch.set_rng_state(rng_data['rng_state'])
        if logger:
            logger.info("Restored torch CPU RNG state.")

    if torch.cuda.is_available() and rng_data.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all(rng_data['cuda_rng_state_all'])
        if logger:
            logger.info("Restored torch CUDA RNG state.")

    if rng_data.get('numpy_rng_state') is not None:
        np.random.set_state(rng_data['numpy_rng_state'])
        if logger:
            logger.info("Restored NumPy RNG state.")

    if rng_data.get('python_rng_state') is not None:
        random.setstate(rng_data['python_rng_state'])
        if logger:
            logger.info("Restored Python RNG state.")

    if generator is not None and 'dataloader_generator_state' in rng_data and rng_data['dataloader_generator_state'] is not None:
        try:
            generator.set_state(rng_data['dataloader_generator_state'])
            if logger:
                logger.info("Restored DataLoader generator state from checkpoint.")
        except Exception as e:
            if logger:
                logger.warning(f"Could not restore DataLoader generator state: {e}")


def restore_scaler_state(
    ckpt_data: Dict[str, Any],
    scaler: Optional[torch.amp.GradScaler] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Restore AMP GradScaler state if present."""
    if scaler is not None and 'scaler_state_dict' in ckpt_data:
        try:
            scaler.load_state_dict(ckpt_data['scaler_state_dict'])
            if logger:
                logger.info("Restored AMP scaler state_dict from checkpoint.")
            return True
        except Exception as e:
            if logger:
                logger.warning(f"Could not restore scaler state_dict: {e}")
    return False
