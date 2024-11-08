"""This is the python file equivalent to the notebook file "metrics_classification.ipynb"""

print("Importing Libraries...")
from itertools import cycle
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from visualdl import vdl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

################################# Change Paths here ########################################################

model_path = r"1.5 dataset wo-Vac_complex_class_512px_-1_rgb_resnext50_32x4d.pt"
data_dir = r"1.5 dataset wo-Vac_complex_class_512px_-1_rgb\valid"
metrics_output_path = r"1.5 dataset wo-Vac_complex_class_512px_-1_rgb\metrics"

############################################################################################################

model = vdl.get_inference_model(model_path, type="classification")
image_size = model.state["custom_data"]["image_size"]


# Initialize variables to hold predictions and labels
y_true = []
y_pred = []


# Dictionary to map subfolder names to numerical labels
class_names = os.listdir(data_dir)
class_mapping = {name: i for i, name in enumerate(class_names)}


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Load images and make predictions
for class_name in tqdm(class_names, desc="Classes"):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in tqdm(os.listdir(class_dir), desc="Images", leave=False):
        img_path = os.path.join(class_dir, img_name)

        img = cv2.resize(
            cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
            (image_size, image_size),
        )
        logits = model.predict([img])[0]  # Assuming model.predict returns logits
        probabilities = softmax(logits)  # Convert logits to probabilities if necessary

        y_true.append(class_mapping[class_name])
        y_pred.append(probabilities)


# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

n_classes = len(class_names)
y_true_binarized = label_binarize(y_true, classes=range(n_classes))


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
if n_classes == 2:
    # Assuming y_pred[:, 1] is the probability of the positive class
    fpr[0], tpr[0], _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
else:
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(
    np.concatenate([fpr[i] for i in range(n_classes)] if n_classes > 2 else [fpr[0]])
)


# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(
        all_fpr, fpr[i if n_classes > 2 else 0], tpr[i if n_classes > 2 else 0]
    )


# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="Micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="Macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
if n_classes == 2:
    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=2,
        label="ROC curve (area = {:.2f})".format(roc_auc[0]),
    )
else:
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {} (area = {:.2f})".format(
                class_names[i], roc_auc[i]
            ),
        )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Plot")
plt.legend(loc="lower right")
plt.savefig("micro_macro_average_roc_curve.png", dpi=500)
plt.close()


for i, class_name in enumerate(class_names):
    if n_classes == 2 and i > 0:
        # Skip the first class in binary classification (usually representing the negative class)
        continue

    fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if n_classes == 2:
        plt.title(f"ROC Curve")
    else:
        plt.title(f"ROC Curve for {class_name}")

    plt.title(f"ROC Curve for {class_name}")
    plt.legend(loc="lower right")
    plt.grid(True)  # Add grid lines for better readability
    plt.tight_layout()  # Adjust the layout to make room for the legend and labels
    figure_name = f"ROC_Curve_for_{class_name}.png"
    plt.savefig(figure_name, dpi=500)
    plt.close()

y_pred_labels = np.argmax(y_pred, axis=1)


# Compute confusion matrix, accuracy, and F1 score
conf_mat = confusion_matrix(y_true, y_pred_labels)
accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels, average="macro")
recall = recall_score(y_true, y_pred_labels, average="macro")
f1 = f1_score(y_true, y_pred_labels, average="macro")
specificity_scores = []
for i in range(n_classes):  # Assuming n_classes is the number of unique classes
    # For each class, calculate specificity
    true_negatives = np.sum(np.delete(np.delete(conf_mat, i, 0), i, 1))
    false_positives = np.sum(np.delete(conf_mat[:, i], i))
    total_actual_negatives = true_negatives + false_positives
    specificity_score = (
        true_negatives / total_actual_negatives if total_actual_negatives != 0 else 0
    )
    specificity_scores.append(specificity_score)


# Macro-average Specificity
specificity = np.mean(specificity_scores)


# Add these lines after the confusion matrix, accuracy, and F1 score calculations
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")


# Plot confusion matrix with annotations
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()  # Adjust layout to make room for the additional text
os.makedirs(metrics_output_path, exist_ok=True)
plt.savefig(os.path.join(metrics_output_path, "confusion.png"), dpi=500)
# plt.show()


# Adjusting the position of annotations
# Increase the y-offset for the annotations if your class_names list is long
y_offset_accuracy = len(class_names) + 0.7  # Adjusted y-offset for accuracy
y_offset_f1 = len(class_names) + 1.0  # Adjusted y-offset for F1 score

precision_offset = len(class_names) + 1.5  # Adjusted y-offset for F1 score
recall_offset = len(class_names) + 2.0  # Adjusted y-offset for F1 score
specifitiy_offset = len(class_names) + 2.5  # Adjusted y-offset for F1 score


with open(os.path.join(metrics_output_path, "averaged_metrics.txt"), "w") as handle:
    handle.write(f"Accuarcy: {accuracy}\n")
    handle.write(f"F1: {f1}\n")
    handle.write(f"Precision: {precision}\n")
    handle.write(f"Recall: {recall}\n")
    handle.write(f"Specifitiy: {specificity}\n")
