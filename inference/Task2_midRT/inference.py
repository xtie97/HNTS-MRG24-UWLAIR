from pathlib import Path
from segmenter import inference 

# define the input and output paths
INPUT_PATH = Path("/input") # these are the paths that Docker will use
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# for testing only 
#INPUT_PATH = Path("test/input")  
#OUTPUT_PATH = Path("test/output")


def run():
    """
    Main function to read input, process the data, and write output.
    """
    
    ### inference ###
    _show_torch_cuda_info()

    # define the inference object
    inf = inference(roi_size=[160, 160, 128], modality="mri", extra_modalities={"label21": "label", "label22": "label"})
    
    # specify the input and output paths
    mid_rt_t2w_head_neck = INPUT_PATH / "images/mid-rt-t2w-head-neck"
    #pre_rt_t2w_head_neck = INPUT_PATH / "images/pre-rt-t2w-head-neck"
    #pre_rt_head_neck_segmentation = INPUT_PATH / "images/pre-rt-head-neck-segmentation"
    #registered_pre_rt_head_neck = INPUT_PATH / "images/registered-pre-rt-head-neck"
    registered_pre_rt_head_neck_segmentation = INPUT_PATH / "images/registered-pre-rt-head-neck-segmentation"

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
    inf.infer_image(ckpt_path_list=ckpt_path_list, 
                    midRT_location=mid_rt_t2w_head_neck,
                    preRT_mask_location=registered_pre_rt_head_neck_segmentation,
                    mask_location=mask_location)
    
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