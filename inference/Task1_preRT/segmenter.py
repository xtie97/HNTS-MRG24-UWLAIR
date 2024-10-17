import os
from typing import Optional
import numpy as np
import torch
from monai.data import decollate_batch, list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    ResampleToMatchd,
    Spacingd,
)
from monai.networks.nets import SegResNetDS
from monai.utils import convert_to_dst_type
from glob import glob
import SimpleITK as sitk
import cc3d

__all__ = ["inference", "load_image_file", "write_mask_file"]


# get the network model
def get_network(in_channels=1, n_class=3):
    model = SegResNetDS(
        init_filters=32,
        blocks_down=[1, 2, 2, 4, 4, 4],
        norm="INSTANCE",
        in_channels=in_channels,
        out_channels=n_class,
        dsdepth=4,
    )
    # input channels: preRT
    return model


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 26  # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


# convert the logits to prediction (apply softmax)
def logits2pred(logits, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return torch.softmax(logits, dim=dim)


# DataTransformBuilder class to build the data transforms for the inference
class DataTransformBuilder:
    def __init__(
        self,
        roi_size: list,
        image_key: str = "image",
        resample: bool = True,
        resample_resolution: Optional[list] = [1.0, 1.0, 1.0],
        modality: str = None,
        extra_modalities: Optional[dict] = None,
    ) -> None:

        self.roi_size, self.image_key = roi_size, image_key
        self.resample, self.resample_resolution = resample, resample_resolution
        if modality is not None:
            self.modality = modality
        self.extra_modalities = extra_modalities if extra_modalities is not None else {}

    def get_load_transforms(self):
        ts = []
        keys = [self.image_key] + list(self.extra_modalities)
        ts.append(
            LoadImaged(
                keys=keys,
                ensure_channel_first=True,
                dtype=None,
                allow_missing_keys=True,
                image_only=True,
            )
        )
        ts.append(
            EnsureTyped(
                keys=keys,
                data_type="tensor",
                dtype=torch.float,
                allow_missing_keys=True,
            )
        )
        return ts

    def threshold_for_mri(self, x):
        # threshold at 0.0 for T2 MRI
        return x > 0.0

    def get_resample_transforms(self, crop_foreground=True):
        ts = []
        keys = [self.image_key]
        extra_keys = self.extra_modalities  # dict
        # self.image_key is PET, self.extra_modalities is CT
        mode = ["bilinear"]
        if crop_foreground:
            ts.append(
                CropForegroundd(
                    keys=keys,
                    source_key=self.image_key,
                    select_fn=self.threshold_for_mri,
                    margin=0,
                    allow_missing_keys=True,
                    allow_smaller=True,
                )
            )  # it can be accomplished in a pre-processing step
        # resample the image to the specified resolution
        if self.resample:
            if self.resample_resolution is None:
                raise ValueError("resample_resolution is not provided")
            pixdim = self.resample_resolution
            ts.append(
                Spacingd(keys=keys, pixdim=pixdim, mode=mode, allow_missing_keys=True)
            )
        # match extra modalities to the key image.
        for extra_key in extra_keys:
            ts.append(
                ResampleToMatchd(
                    keys=extra_key, key_dst=self.image_key, dtype=np.float32
                )
            )
        return ts

    def get_normalize_transforms(self):
        ts = []
        modalities = {self.image_key: self.modality}
        modalities.update(self.extra_modalities)

        for key, normalize_mode in modalities.items():
            normalize_mode = normalize_mode.lower()
            if normalize_mode in ["mri"]:
                ts.append(
                    NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True)
                )
            else:
                raise ValueError(
                    "Unsupported normalize_mode" + str(self.normalize_mode)
                )

        if len(self.extra_modalities) > 0:
            ts.append(
                ConcatItemsd(keys=list(modalities), name=self.image_key)
            )  # concatenate all modalities at the channels
            ts.append(DeleteItemsd(keys=list(self.extra_modalities)))  # release memory
        return ts

    @classmethod
    def get_postprocess_transform(cls, invert=False, transform=None) -> Compose:
        ts = []
        if invert and transform is not None:
            ts.append(
                Invertd(
                    keys="pred",
                    orig_keys="image",
                    transform=transform,
                    nearest_interp=False,
                )
            )

        return Compose(ts)

    def __call__(self) -> Compose:

        ts = []
        ts.extend(self.get_load_transforms())
        ts.extend(self.get_resample_transforms())
        ts.extend(self.get_normalize_transforms())

        return Compose(ts)


class inference:
    def __init__(
        self,
        roi_size: list,
        modality: str = None,
        extra_modalities: Optional[dict] = None,
    ) -> None:
        # specify the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the model
        model = get_network()
        model = model.to(self.device)
        self.model = model

        # number of classes is always 3 (bg, GTVp, GTVn)
        self.n_class = 3
        # Sliding window inference
        self.sliding_inferrer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=2,
            overlap=0.625,
            mode="gaussian",
            cache_roi_weight_map=True,
            progress=True,
            cpu_thresh=512**3 // self.n_class,
        )

        self.data_tranform_builder = DataTransformBuilder(
            roi_size=roi_size,
            modality=modality,
            extra_modalities=extra_modalities,
        )
        self.inf_transform = self.data_tranform_builder()
        self.post_transforms = DataTransformBuilder.get_postprocess_transform(
            invert=True, transform=self.inf_transform
        )

    def checkpoint_load(self, ckpt: str, model: torch.nn.Module):

        if not os.path.isfile(ckpt):
            raise ValueError("Invalid checkpoint file" + str(ckpt))
        else:
            checkpoint = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)

    @torch.no_grad()
    def infer_image(
        self, ckpt_path_list: list, image_location: str, mask_location: str
    ) -> None:
        self.model.eval()

        image_file = load_image_file(location=image_location)
        batch_data = self.inf_transform({"image": image_file})
        batch_data = list_data_collate([batch_data])

        data = batch_data["image"].as_subclass(torch.Tensor).to(self.device)

        pred = torch.zeros(
            data.shape[0], self.n_class, *data.shape[2:], device=self.device
        )
        for ckpt_path in ckpt_path_list:
            self.checkpoint_load(ckpt=ckpt_path, model=self.model)
            logits = self.sliding_inferrer(inputs=data, network=self.model)
            pred += logits2pred(logits=logits.float())

        pred /= len(ckpt_path_list)
        batch_data["pred"] = convert_to_dst_type(
            pred, batch_data["image"], dtype=pred.dtype, device=pred.device
        )[
            0
        ]  # [0] is output
        # Convert source data to the same data type and device as the destination data

        pred = torch.stack(
            [self.post_transforms(x)["pred"] for x in decollate_batch(batch_data)]
        )
        pred = pred.argmax(dim=1, keepdim=True).squeeze()
        pred = torch.permute(pred, (2, 1, 0))
        # convert to numpy array
        pred = pred.detach().cpu().numpy()
        write_mask_file(mask_location, pred, image_file)


def load_image_file(location):
    # input_files = glob(str(location / "*.mha")) # Grand Challenge uses MHA files
    input_files = glob(str(location / "*.mha"))  # Grand Challenge uses MHA files
    return input_files[0]


def write_mask_file(location: str, segmentation: np.ndarray, input_file: str):
    location.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(input_file)
    # get the pixel dimension of segmentation image
    voxel_size = image.GetSpacing()
    # print(f"Voxel size: {voxel_size}")
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    # print(f"Voxel: {voxel_volume}")

    # post-processing, remove small connected components
    pred_class1 = np.where(segmentation == 1, 1, 0)
    pred_class1 = con_comp(pred_class1)
    for i in range(1, np.max(pred_class1) + 1):
        if np.sum(pred_class1 == i) * voxel_volume < 500:  # mm^3
            pred_class1[pred_class1 == i] = 0
    pred_class1[pred_class1 > 0] = 1

    pred_class2 = np.where(segmentation == 2, 1, 0)
    pred_class2 = con_comp(pred_class2)
    for i in range(1, np.max(pred_class2) + 1):
        if np.sum(pred_class2 == i) * voxel_volume < 500:  # mm^3
            pred_class2[pred_class2 == i] = 0
    pred_class2[pred_class2 > 0] = 2

    segmentation = pred_class1 + pred_class2

    # convert the numpy array back to a SimpleITK image
    segmentation_image = sitk.GetImageFromArray(segmentation)

    segmentation_image.CopyInformation(image)
    # Cast the segmentation image to 8-bit unsigned int
    segmentation_image = sitk.Cast(segmentation_image, sitk.sitkUInt8)
    # Write to a MHA file
    suffix = ".mha"
    sitk.WriteImage(
        segmentation_image, location / f"output{suffix}", useCompression=True
    )
