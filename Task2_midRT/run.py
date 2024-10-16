import os
from monai.utils import optional_import
from src.segmenter import main, run_segmenter

if __name__ == "__main__":
    # specify the visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    fire, fire_is_imported = optional_import("fire")
    if fire_is_imported:
        fire.Fire(main)
