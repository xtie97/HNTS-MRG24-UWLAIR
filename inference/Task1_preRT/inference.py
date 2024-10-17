from pathlib import Path
from segmenter import inference

# define the input and output paths
INPUT_PATH = Path("/input")  # these are the paths that Docker will use
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# for testing only
# INPUT_PATH = Path("test/input")
# OUTPUT_PATH = Path("test/output")


def run():
    """
    Main function to read input, process the data, and write output.
    """

    ### inference ###
    _show_torch_cuda_info()

    # define the inference object
    inf = inference(roi_size=[192, 192, 128], modality="mri")

    # specify the input and output paths
    image_location = INPUT_PATH / "images/pre-rt-t2w-head-neck/"
    mask_location = OUTPUT_PATH / "images/mri-head-neck-segmentation"
    # specify the checkpoint paths
    ckpt_path_list = [
        RESOURCE_PATH / "model_0.pt",
        RESOURCE_PATH / "model_1.pt",
        RESOURCE_PATH / "model_2.pt",
        RESOURCE_PATH / "model_3.pt",
        RESOURCE_PATH / "model_4.pt",
        RESOURCE_PATH / "model_5.pt",
        RESOURCE_PATH / "model_6.pt",
        RESOURCE_PATH / "model_7.pt",
        RESOURCE_PATH / "model_8.pt",
        RESOURCE_PATH / "model_9.pt",
    ]

    # run inference
    inf.infer_image(
        ckpt_path_list=ckpt_path_list,
        image_location=image_location,
        mask_location=mask_location,
    )

    #################
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
