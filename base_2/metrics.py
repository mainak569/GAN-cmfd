import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    jaccard_score
)
from skimage.measure import label


# ============================================================
# PIXEL-LEVEL METRICS
# ============================================================

def compute_pixel_metrics(gt_mask, pred_mask):
    """
    Computes pixel-level binary segmentation metrics.

    Parameters:
        gt_mask   : numpy array (H, W) -> {0,1}
        pred_mask : numpy array (H, W) -> {0,1}

    Returns:
        dict of pixel-level metrics
    """

    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)

    return {
        "Pixel_Precision": precision,
        "Pixel_Recall": recall,
        "Pixel_F1": f1,
        "Pixel_Accuracy": accuracy,
        "Pixel_IoU": iou
    }


# ============================================================
# REGION-LEVEL METRICS
# ============================================================

def compute_region_metrics(gt_mask, pred_mask):

    gt_labeled = label(gt_mask)
    pred_labeled = label(pred_mask)

    gt_regions = np.unique(gt_labeled)
    pred_regions = np.unique(pred_labeled)

    gt_regions = gt_regions[gt_regions != 0]
    pred_regions = pred_regions[pred_regions != 0]

    # Case 1: no GT and no prediction → perfect
    if len(gt_regions) == 0 and len(pred_regions) == 0:
        return {"Region_mIoU": 1.0, "Region_IoUs": []}

    # Case 2: no GT but prediction exists → false alarm
    if len(gt_regions) == 0 and len(pred_regions) > 0:
        return {"Region_mIoU": 0.0, "Region_IoUs": []}

    ious = []
    used_pred = set()

    for gt_id in gt_regions:
        gt_region = (gt_labeled == gt_id)
        best_iou = 0
        best_pred = None

        for pred_id in pred_regions:
            if pred_id in used_pred:
                continue

            pred_region = (pred_labeled == pred_id)

            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()

            if union == 0:
                continue

            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_pred = pred_id

        if best_pred is not None:
            used_pred.add(best_pred)

        ious.append(best_iou)

    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0

    return {
        "Region_mIoU": mean_iou,
        "Region_IoUs": ious
    }



# ============================================================
# 3️⃣ DATASET-LEVEL AGGREGATION
# ============================================================

def aggregate_pixel_metrics(metrics_list):
    """
    Averages pixel-level metrics across dataset
    """

    keys = metrics_list[0].keys()
    aggregated = {}

    for key in keys:
        aggregated[key] = np.mean([m[key] for m in metrics_list])

    return aggregated


def aggregate_region_metrics(metrics_list):
    """
    Averages region-level metrics across dataset
    """

    mean_ious = [m["Region_mIoU"] for m in metrics_list]

    return {
        "Region_mIoU": np.mean(mean_ious)
    }

# ============================================================
# DATASET-LEVEL EVALUATION (MICRO-AVERAGE FOR PIXELS)
# ============================================================

def evaluate_segmentation(gt_masks, pred_masks):
    """
    Evaluate full dataset with:
    - Pixel metrics computed globally (micro average)
    - Region metrics averaged per image
    """

    all_gt = []
    all_pred = []

    region_metrics_all = []

    for gt, pred in zip(gt_masks, pred_masks):

        gt = gt.astype(np.uint8)
        pred = pred.astype(np.uint8)

        all_gt.append(gt.flatten())
        all_pred.append(pred.flatten())

        region_metrics_all.append(compute_region_metrics(gt, pred))

    # -------- Pixel-level (global) --------
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    pixel_results = compute_pixel_metrics(all_gt, all_pred)

    # -------- Region-level (image-wise average) --------
    mean_ious = [m["Region_mIoU"] for m in region_metrics_all]
    region_results = {
        "Region_mIoU": np.mean(mean_ious)
    }

    return {
        **pixel_results,
        **region_results
    }
