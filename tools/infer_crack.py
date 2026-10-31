#!/usr/bin/env python3
"""Run canonical SAGE-Lite B2 inference on one image or a directory."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from sage.networks import create_b2_unet

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def build_model(config, checkpoint_path, device):
    model_type = config.get("model", "B2")
    if model_type not in {"B2", "SAGE_LITE", "ConvNeXt_Transformer_UNet"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = create_b2_unet(
        num_transformer_layers=int(config.get("num_transformer_layers", 12)),
        img_size=int(config.get("img_size", 448)),
        pretrained=False,
        sage_config=config.get("sage_config", {}),
        p3_mode=config.get("p3_mode"),
        use_plu_head=config.get("use_plu_head", False),
        use_s2_gate=config.get("use_s2_gate", False),
        s2_gate_kernel_size=config.get("s2_gate_kernel_size", 3),
        use_s2_gate_block2=config.get("use_s2_gate_block2", False),
        s2_gate_block2_kernel_size=config.get("s2_gate_block2_kernel_size", 3),
        s2_gate_block2_dilation=config.get("s2_gate_block2_dilation", 1),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_full_image_direct(model, image, device, target_size=448):
    """Mirror the official evaluator's preprocessing and prediction path."""
    height, width = image.shape[:2]
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - mean) / std
    is_gtx16xx = False
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        is_gtx16xx = "1650" in name or "1660" in name
    with torch.no_grad():
        if device.type == "cuda" and not is_gtx16xx:
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)
    logits_np = logits.squeeze().float().cpu().numpy()
    return cv2.resize(logits_np, (width, height), interpolation=cv2.INTER_LINEAR)

def collect_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output-dir", required=True, help="Directory for binary PNG masks")
    parser.add_argument("--config", required=True, help="B2 YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Trained B2 checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = build_model(config, args.checkpoint, device)
    images = collect_images(args.input)
    if not images:
        raise RuntimeError(f"No supported images found in: {args.input}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logits = predict_full_image_direct(model, image, device, target_size=int(config.get("img_size", 448)))
        mask = (logits > 0.0).astype(np.uint8) * 255
        output_path = output_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Could not write output: {output_path}")
        print(f"{image_path} -> {output_path}")

if __name__ == "__main__":
    main()
