"""
Trainers Module

Contains different training strategies:
- Step 1: Gradient Descent (baseline, in models/unimol_wrapper.py)
- Step 2: EGGROLL with MLP head (step2_eggroll.py)
- Step 3: EGGROLL + GP Co-evolution (step3_eggroll_gp.py)
"""

from .step2_eggroll import Step2Trainer, Step2Config, UniMolEGGROLLWrapper
from .step3_eggroll_gp import Step3Trainer, Step3Config, run_step3, EVOGP_AVAILABLE

__all__ = [
    # Step 2
    'Step2Trainer', 'Step2Config', 'UniMolEGGROLLWrapper',
    # Step 3
    'Step3Trainer', 'Step3Config', 'run_step3', 'EVOGP_AVAILABLE'
]