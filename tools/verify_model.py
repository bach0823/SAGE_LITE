import os
import sys
import logging
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_verification():
    print("=" * 60)
    print("SAGE-Lite Repository & Architecture Verification")
    print("=" * 60)
    
    try:
        # 1. Test Imports
        print("\n[1/4] Testing Core Imports...")
        from sage.networks import create_b2_unet
        from sage.components.sage_layer import SageLayer
        from sage.networks.s2_gate import S2GateModule
        print("[OK] Successfully imported SAGE-Lite components and factory.")

        # 2. Test Model Instantiation (SAGE-Lite B2 Configuration)
        print("\n[2/4] Instantiating SAGE-Lite B2 Model...")
        sage_config = {
            'top_k': 2,
            'gating_type': 'sigmoid',
            'shared_expert_indices': [0, 1, 2, 3],
            'router_hidden_dim': 64,
            'load_balance_factor': 0.01,
            'logit_modulation': True,
            'expert_dropout': 0.1,
            'fusion_type': 'residual',
            'residual_scale': 0.1
        }
        
        model = create_b2_unet(
            num_classes=1,
            img_size=448,
            num_transformer_layers=4,
            pretrained=False,
            sage_config=sage_config,
            use_s2_gate=True,
            s2_gate_kernel_size=3,
            use_s2_gate_block2=True,
            s2_gate_block2_kernel_size=3,
            s2_gate_block2_dilation=2,
        )
        print("[OK] SAGE-Lite B2 instantiated successfully.")
        
        # 3. Model Information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
        print(f"  - Input Size: 448x448")
        
        # 4. Forward Pass Test
        print("\n[3/4] Testing Forward Pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(2, 3, 448, 448).to(device)
        print(f"  - Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print("[OK] Forward pass successful.")
        print(f"  - Output shape: {output.shape}")
        
        # 5. Check SAGE & S2-Gate components
        print("\n[4/4] Verifying SAGE Layers & S2-Gate Injections...")
        sage_layers = [m for m in model.modules() if isinstance(m, SageLayer)]
        s2_gates = [m for m in model.modules() if isinstance(m, S2GateModule)]
        print(f"[OK] Found {len(sage_layers)} SageLayers injected.")
        print(f"[OK] Found {len(s2_gates)} S2GateModules active.")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] VERIFICATION SUCCESSFUL: Repository is functional!")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "!" * 60)
        print(f"[FAILED] VERIFICATION FAILED: {str(e)}")
        print("!" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
