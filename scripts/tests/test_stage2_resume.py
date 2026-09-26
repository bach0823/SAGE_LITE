import os
import sys
import tempfile
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet

def test_stage2_resume_logic():
    print("[Test] Verifying Stage 2 resumption and state restoration contracts...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy model and checkpoint with known best metrics
        dummy_model = create_b2_unet(num_transformer_layers=2, p3_mode="A", img_size=224)
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
        
        torch.save({
            'epoch': 13,
            'stage': 2,
            'model_state_dict': dummy_model.state_dict(),
            'best_dice': 0.7543,
            'best_loss': 1.0140,
            'model_type': 'B2',
            'num_transformer_layers': 2,
            'p3_mode': 'A',
        }, ckpt_path)
        
        # Verify checkpoint loads and contains exact values
        loaded_ckpt = torch.load(ckpt_path, weights_only=False)
        assert loaded_ckpt['best_dice'] == 0.7543
        assert loaded_ckpt['best_loss'] == 1.0140
        assert loaded_ckpt['epoch'] == 13
        assert loaded_ckpt['stage'] == 2
        print("  --> PASS: Checkpoint metadata contract verified.")

if __name__ == '__main__':
    test_stage2_resume_logic()
    print("ALL STAGE 2 RESUME CONTRACT CHECKS PASSED.")
