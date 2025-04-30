import json

# file_name = "pred.v2.final.json"
file_name = "pred.final.json"
# predictions_file = "annotations_v2.json"
predictions_file = "pred-annotations.json"
with open(predictions_file, "r") as f:
    predictions = json.load(f)

prediction_images = {item["id"]: item["file_name"] for item in predictions["images"]}
prediction_annotations = []
for item in predictions["annotations"]:
    if "file_name" not in item:
        item["file_name"] = prediction_images[item["image_id"]]
    prediction_annotations.append(item)

predictions["annotations"] = prediction_annotations
with open(file_name, "w") as json_file:
    json.dump(predictions, json_file, indent=4)
