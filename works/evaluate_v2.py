import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def convert_predictions(predictions, coco_gt):
    converted_predictions = []
    for annotation in predictions["annotations"]:
        file_name = annotation["file_name"]
        img_id = -1
        for img in coco_gt.dataset["images"]:
            if img["file_name"] == file_name:
                img_id = img["id"]
                break
        assert img_id != -1, f"Image id not found for file_name: {file_name}"
        converted_predictions.append({
            "image_id": img_id,
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "score": annotation["score"]
        })
    return converted_predictions

def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Load ground truth and predictions
ground_truth_annotations_file = 'test_labels_200_with_iscrowd.json'
coco_gt = COCO(ground_truth_annotations_file)
# predictions_file = 'annotations_v2.json'
predictions_file = 'v1  -epoch_10_threshold=0.1.json'
# predictions_file = 'v4-epoch_10_threshold=0.1.json'
# predictions_file = 'small-epoch_10_threshold=0.1.json'
# predictions_file = 'v2-epoch_10_threshold=0.1.json'
# predictions_file = 'epoch_10_threshold=0.1.json'
with open(predictions_file, 'r') as f:
    predictions = json.load(f)

# Convert predictions
converted_predictions = convert_predictions(predictions, coco_gt)
coco_dt = coco_gt.loadRes(converted_predictions)

# Evaluate using pycocotools
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
mAP = coco_eval.stats[0]
print(f"Mean AP: {mAP}")

# Compute Mean IoU
ious = []
for gt_ann in coco_gt.dataset['annotations']:
    image_id = gt_ann['image_id']
    gt_bbox = gt_ann['bbox']
    pred_anns = [ann for ann in converted_predictions if ann['image_id'] == image_id]
    max_iou = 0
    for pred_ann in pred_anns:
        pred_bbox = pred_ann['bbox']
        iou = bbox_iou(gt_bbox, pred_bbox)
        max_iou = max(max_iou, iou)
    if max_iou > 0:
        ious.append(max_iou)
mean_iou = np.mean(ious) if ious else 0
print(f"Mean IoU: {mean_iou}")