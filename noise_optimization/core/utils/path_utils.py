import os
import sys

def setup_proteina_path():
    """Add proteina directory to sys.path for proteinfoundation imports.
    
    This handles the directory structure where 'proteina' is a sibling to 'core'.
    """
    current_file = os.path.abspath(__file__)
    # utils -> core -> noise_optimization
    noise_opt_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    proteina_dir = os.path.join(noise_opt_dir, "proteina")
    if os.path.exists(proteina_dir) and proteina_dir not in sys.path:
        sys.path.insert(0, proteina_dir)
    return proteina_dir

