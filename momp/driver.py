"""
ROMP Main Entry Script
----------------------
This script serves as the primary execution point for the Rainy season Onset
Metrics Package (ROMP). It orchestrates the calculation of skill scores
or the generation of spatial performance maps.

Usage:
    momp-run
    or
    python -m momp.driver
"""

import logging
import os
import sys
import tempfile
import traceback

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "momp_mplconfig"))

# Local Package Imports
#from momp.lib.loader import cfg, setting
from momp.lib.loader import get_cfg, get_setting
from momp.utils.printing import print_momp_banner

# Create a logs directory if it doesn't exist
log_dir = "logs"
#os.makedirs(log_dir, exist_ok=True)

# Configure logging for professional status updates
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        #logging.FileHandler(f"{log_dir}/momp_run.log"), # Saves to file
        logging.StreamHandler()                        # Prints to terminal
    ]
)

logger = logging.getLogger(__name__)

cfg, setting = get_cfg(), get_setting()

#def run_momp(cfg=get_cfg(), setting=get_setting()):
def run_momp(cfg=cfg, setting=setting):
    """
    Executes the standard ROMP evaluation workflow.

    This function triggers the bin-based skill score calculations
    or generates spatial maps for False Alarm Ratio (FAR),
    Miss Rate (MR), and Mean Absolute Error (MAE).

    Args:
        cfg: The loaded configuration dictionary/SimpleNameSpace object.
        setting: The environment and directory settings.
    """

    print_momp_banner(cfg)

#    print("\n\n\n cfg.max_forecast_day = ", cfg.max_forecast_day)
#    print("\n\n\n cfg.model = ", cfg.model_list)

    logger.info("Starting ROMP Workflow...")

    try:
        if cfg.workflow in ("onset", "all"):
            from momp.app.bin_skill_score import skill_score_in_bins

            if cfg.probabilistic:
                # Calculate and save probabilistic skill scores in defined day bins.
                skill_score_in_bins()
            else:
                from momp.app.spatial_far_mr_mae import spatial_far_mr_mae_map

                # Generate deterministic spatial metrics and maps.
                spatial_far_mr_mae_map()

        if cfg.workflow in ("cra", "all"):
            from momp.app.cra_run import run_cra_workflow

            logger.info("Running CRA rainfall verification...")
            run_cra_workflow(cfg, setting)

        logger.info("ROMP Workflow completed successfully!")

    except Exception as e:
        logger.error(f"ROMP failed during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

# ------------------------------------------------------------------------------
# EXECUTION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    run_momp()
