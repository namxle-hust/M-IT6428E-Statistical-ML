import json
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Load the JSON files
with open("test_labels_200.json", "r") as f:
    test_labels = json.load(f)
    test_images = {item["id"]: item["file_name"] for item in test_labels["images"]}


with open("pred-annotations.json", "r") as f:
    predictions = json.load(f)
    prediction_images = {
        item["id"]: item["file_name"] for item in predictions["images"]
    }

# Create dictionaries for easy lookup by image_id
true_dict = {
    test_images[item["image_id"]]: item["category_id"]
    for item in test_labels["annotations"]
}
pred_dict = {
    prediction_images[item["image_id"]]: item["category_id"]
    for item in predictions["annotations"]
}

# Initialize variables to calculate metrics
y_true = []
y_pred = []

# Compare predictions and ground truth
for image in true_dict:
    if image in pred_dict:
        y_true.append(true_dict[image])
        y_pred.append(pred_dict[image])

# Calculate confusion matrix components
cm = confusion_matrix(y_true, y_pred)

# Calculate TP, TN, FP, FN for each class
tp = cm.diagonal()
fn = cm.sum(axis=1) - tp
fp = cm.sum(axis=0) - tp
tn = cm.sum() - (tp + fn + fp)

# Calculate Label Accuracy and Bag Accuracy
label_accuracy = accuracy_score(y_true, y_pred)
bag_accuracy = (tp + tn) / (tp + tn + fp + fn)

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average="weighted")

# Print the results
print(f"Label Accuracy: {label_accuracy}")
print(f"Bag Accuracy: {bag_accuracy}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"F1 Score: {f1}")
