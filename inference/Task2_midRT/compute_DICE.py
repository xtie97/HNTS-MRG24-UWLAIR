import os
import numpy as np
import nibabel as nib
from tqdm import tqdm 
from multiprocessing import Process
from pathlib import Path

try:
  import SimpleITK as sitk
except:
  os.system('pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install SimpleITK --upgrade')
  import SimpleITK as sitk

# for testing
INPUT_PATH = Path("test/input/images/pre-rt-t2w-head-neck/")  
OUTPUT_PATH = Path("test/output/images/mri-head-neck-segmentation")
GROUND_TRUTH_PATH = Path("groundtruth")

def compute_volumes(im):
    """
    Compute the volumes of the GTVp and the GTVn
    """
    spacing = im.GetSpacing()
    voxvol = spacing[0] * spacing[1] * spacing[2]
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(im, im)
    nvoxels1 = stats.GetCount(1)
    nvoxels2 = stats.GetCount(2)
    return nvoxels1 * voxvol, nvoxels2 * voxvol

def compute_agg_dice(intermediate_results):
    """
    Compute the aggregate dice score from the intermediate results
    """
    aggregate_results = {}
    TP1s = [v["TP1"] for v in intermediate_results]
    TP2s = [v["TP2"] for v in intermediate_results]
    vol_sum1s = [v["vol_sum1"] for v in intermediate_results]
    vol_sum2s = [v["vol_sum2"] for v in intermediate_results]
    DSCagg1 = 2 * np.sum(TP1s) / np.sum(vol_sum1s)
    DSCagg2 = 2 * np.sum(TP2s) / np.sum(vol_sum2s)
    aggregate_results['AggregatedDsc'] = {
        'GTVp': DSCagg1,
        'GTVn': DSCagg2,
        'mean': np.mean((DSCagg1, DSCagg2)),
    }
    return aggregate_results

def get_intermediate_metrics(patient_ID, groundtruth, prediction):
    """
    Compute intermediate metrics for a given groundtruth and prediction.
    These metrics are used to compute the aggregate dice.
    """
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(groundtruth, prediction)

    DSC1 = overlap_measures.GetDiceCoefficient(1)
    DSC2 = overlap_measures.GetDiceCoefficient(2)

    vol_gt1, vol_gt2 = compute_volumes(groundtruth)
    vol_pred1, vol_pred2 = compute_volumes(prediction)

    vol_sum1 = vol_gt1 + vol_pred1
    vol_sum2 = vol_gt2 + vol_pred2
    TP1 = DSC1 * (vol_sum1) / 2
    TP2 = DSC2 * (vol_sum2) / 2
    return {
        "PatientID": patient_ID, # added patient ID so we can pinpoint exact results if needed
        "TP1": TP1,
        "TP2": TP2,
        "vol_sum1": vol_sum1,
        "vol_sum2": vol_sum2,
        "DSC1": DSC1,
        "DSC2": DSC2,
        "vol_gt1": vol_gt1, # needed if you want to exclude empty ground truths in conventional DSC calcs
        "vol_gt2": vol_gt2, 
    }

def resample_prediction(groundtruth, prediction):
    """
    Resample the prediction to the groundtruth physical domain
    """
    resample = sitk.ResampleImageFilter()
    resample.SetSize(groundtruth.GetSize())
    resample.SetOutputDirection(groundtruth.GetDirection())
    resample.SetOutputOrigin(groundtruth.GetOrigin())
    resample.SetOutputSpacing(groundtruth.GetSpacing())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(prediction) 

def check_prediction(patient_ID, groundtruth, prediction):
    """
    Check if the prediction is valid and apply padding if needed
    """

    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    groundtruth = caster.Execute(groundtruth)
    prediction = caster.Execute(prediction)

    # Check labels
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(prediction, prediction)
    labels = stats.GetLabels()
    if not all([l in [0, 1, 2] for l in labels]):
        raise RuntimeError(
            f"Patient {patient_ID}: Error. The labels are incorrect. The labels should be background: 0, GTVp: 1, GTVn: 2."
        )
    # Check spacings
    if not np.allclose(
            groundtruth.GetSpacing(), prediction.GetSpacing(), atol=0.000001):
        raise RuntimeError(
            f"Patient {patient_ID}: Error. The resolution of the prediction is different from the MRI ground truth resolution."
        )

    # Check if resampling is needed
    needs_resampling = False
    if prediction.GetSize() != groundtruth.GetSize():
        needs_resampling = True
    elif not np.allclose(prediction.GetDirection(), groundtruth.GetDirection(), atol=0.000001):
        needs_resampling = True
    elif not np.allclose(prediction.GetOrigin(), groundtruth.GetOrigin(), atol=0.000001):
        needs_resampling = True

    if needs_resampling:
        print(f"Patient {patient_ID}: Prediction checked, resampling prediction to match ground truth...")
        prediction = resample_prediction(groundtruth, prediction)
    else:
        print(f'Patient {patient_ID}: Prediction checked, everything correct and no resampling needed.')
        # To be sure that sitk won't trigger unnecessary errors
        prediction.SetSpacing(groundtruth.GetSpacing())

    return prediction


if __name__ == "__main__":
  # DSCagg Calculation
  # first set up the ground truth and prediction paths
  
  
  # get the ground truth files
  groundtruth_files = [os.path.join(GROUND_TRUTH_PATH, file) for file in os.listdir(GROUND_TRUTH_PATH) if ".nii.gz" in file]
  prediction_files = [os.path.join(OUTPUT_PATH, file) for file in os.listdir(OUTPUT_PATH) if ".mha" in file]
  
  results = list()
  for patient_id in range(len(groundtruth_files)):
    if len(groundtruth_files) != 1 or len(prediction_files) != 1:
        raise ValueError(f"Error: {gt_file} or {pred_file} does not exist or is not unique.")
    gt_file = groundtruth_files[0]
    pred_file = prediction_files[0]
    prediction = sitk.ReadImage(pred_file)
    groundtruth = sitk.ReadImage(gt_file)
    prediction = check_prediction(patient_id, groundtruth, prediction) 
    results.append(get_intermediate_metrics(patient_id, groundtruth, prediction))

  # Compute and display aggregate dice scores
  # set a line break
  print("\n")
  print("-"*66)
  print("\n")

  agg_dice_scores = compute_agg_dice(results)
  print(f"Aggregate dice scores: {agg_dice_scores}\n")
 