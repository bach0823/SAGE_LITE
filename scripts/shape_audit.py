import os
import sys
import torch

# Add project root to sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b0_unet, create_b1_unet

def audit():
    print("=" * 60)
    print("=== SAGE-Lite Shape Audit & Preflight Verification ===")
    print("=" * 60)

    batch_size = 2
    img_size = 448
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"\n[0] Input Specification:")
    print(f"  Input Tensor Shape: {tuple(x.shape)} (B={batch_size}, C=3, H={img_size}, W={img_size})")

    # ==========================================
    # 1. Audit Baseline B0 (Pure ConvNeXtV2)
    # ==========================================
    print("\n" + "=" * 50)
    print("=== [1] Auditing Baseline B0 (Pure ConvNeXtV2-Femto U-Net) ===")
    print("=" * 50)
    
    b0_model = create_b0_unet(pretrained=True)
    b0_info = b0_model.get_model_info()
    
    print(f"  Model Name: {b0_info['model_name']}")
    print(f"  Total Parameters: {b0_info['total_parameters']:,}")
    
    # Check for ViT presence
    has_vit = any('vit' in str(type(m)).lower() or 'transformer' in str(type(m)).lower() for m in b0_model.modules())
    print(f"  Contains ViT/Transformer blocks? {'YES (FAIL)' if has_vit else 'NO (PASS)'}")
    assert not has_vit, "B0 should not contain ViT!"
    
    # Forward Pass Audit
    print("\n  [B0] Architecture Forward Pass:")
    features = b0_model.backbone(x)
    print("    ConvNeXt Multi-Scale Features (Skips + Bottleneck):")
    for i, feat in enumerate(features):
        stage_name = "Bottleneck" if i == len(features) - 1 else f"Skip {i}"
        print(f"      {stage_name}: {tuple(feat.shape)}")
        
    logits_b0 = b0_model(x)
    print(f"    Final Logits Output Shape: {tuple(logits_b0.shape)}")
    assert logits_b0.shape == (batch_size, 1, img_size, img_size)

    # ==========================================
    # 2. Audit Baseline B1 (Hybrid ConvNeXtV2 + ViT)
    # ==========================================
    print("\n" + "=" * 50)
    print("=== [2] Auditing Baseline B1 (ConvNeXtV2-Femto + ViT Late Fusion) ===")
    print("=" * 50)
    
    b1_model = create_b1_unet(pretrained=True)
    b1_info = b1_model.get_model_info()
    
    print(f"  Model Name: {b1_info['model_name']}")
    print(f"  Total Parameters: {b1_info['total_parameters']:,}")
    
    # Check for ViT presence
    has_vit_b1 = any('vit' in str(type(m)).lower() or 'transformer' in str(type(m)).lower() for m in b1_model.modules())
    print(f"  Contains ViT/Transformer blocks? {'YES (PASS)' if has_vit_b1 else 'NO (FAIL)'}")
    assert has_vit_b1, "B1 MUST contain ViT blocks!"
    
    # Check ViT Blocks specifically inside backbone
    num_vit_blocks = len(b1_model.backbone.transformer_blocks) if hasattr(b1_model.backbone, 'transformer_blocks') else 0
    print(f"  Configured Transformer Blocks count: {num_vit_blocks}")
    
    # Forward Pass Audit
    print("\n  [B1] Architecture Forward Pass:")
    feat_dict = b1_model.backbone(x)
    print("    Hybrid Backbone Features:")
    for i, skip in enumerate(feat_dict['skips']):
        print(f"      Skip {i} (ConvNeXt): {tuple(skip.shape)}")
    print(f"      Bottleneck (ViT Output mapped to spatial grid): {tuple(feat_dict['bottleneck'].shape)}")
    
    logits_b1 = b1_model(x)
    print(f"    Final Logits Output Shape: {tuple(logits_b1.shape)}")
    assert logits_b1.shape == (batch_size, 1, img_size, img_size)

    print("\n" + "=" * 60)
    print("=== All Architecture Shape Audits Passed Successfully ===")
    print("=" * 60)

if __name__ == '__main__':
    audit()
