"""This is the python file equivalent to the notebook file "metrics_segmentation.ipynb"""


print("Importing Libraries...")

# General Setup
import os

# Selection of Graphical Processing Unit (GPU) for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "0"  # Only nVidia GPUs are counted, not integrated GPUs

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from visualdl import vdl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

################################# Change Paths here ########################################################

inference_model = vdl.get_inference_model(
    r"1.5 dataset Hyphe_seg_Hyphe_i-fi_-1_rgb_tu-resnest50d, Unet.pt"
)
validation_images_path = r"1.5 dataset Hyphe_seg_Hyphe_i-fi_-1_rgb\valid\images"
validation_labels_path = r"1.5 dataset Hyphe_seg_Hyphe_i-fi_-1_rgb\valid\labels"
metrics_output_dir = r"1.5 dataset Hyphe_seg_Hyphe_i-fi_-1_rgb\metrics"

############################################################################################################

os.makedirs(metrics_output_dir, exist_ok=True)


## Create metrics with visualdl models segmentation
def simple_inference(model, image):
    """Resize image to 1024x1024 and run inference"""
    # Resize image to 1024x1024
    resized_image = cv2.resize(image, (2048, 2048))

    # Get prediction
    prediction = model.predict([resized_image], single_class_per_contour=False)
    prediction = prediction[0][0]
    # Resize prediction back to original size
    original_h, original_w = image.shape[:2]
    final_pred = cv2.resize(
        prediction.astype(np.uint8),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return final_pred


def create_overlay(original_img, mask, alpha=0.5, color=[255, 0, 0]):
    """Create an overlay of the mask on the original image"""
    overlay = np.zeros_like(original_img)
    overlay[mask > 0] = color  # Apply the specified color for positive regions
    return cv2.addWeighted(original_img, 1 - alpha, overlay, alpha, 0)


def calculate_iou(pred, target):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics and return as dictionary"""
    # Convert to binary (0 or 1)
    y_true_bin = y_true > 0
    y_pred_bin = y_pred > 0

    # Flatten arrays
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()

    # Calculate metrics
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    iou = calculate_iou(y_pred_bin, y_true_bin)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)

    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp + 1e-6)

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "Specificity": specificity,
        "Confusion Matrix": cm,
    }


# Initialize aggregated metrics
all_metrics = []

# Create metric output files
metrics_file = os.path.join(metrics_output_dir, "metrics.txt")
aggregate_metrics_file = os.path.join(metrics_output_dir, "aggregate_metrics.txt")

# Main execution
for file in tqdm(os.listdir(validation_images_path)):
    image_path = os.path.join(validation_images_path, file)
    label_path = os.path.join(validation_labels_path, file)

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.imread(image_path)

    # Use simple inference instead of sliding window
    preds = simple_inference(inference_model, img)

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    metrics = calculate_metrics(label, preds)
    all_metrics.append(metrics)

    plot_confusion_matrix(
        metrics["Confusion Matrix"],
        os.path.join(
            metrics_output_dir, f"confusion_matrix_{os.path.splitext(file)[0]}.png"
        ),
    )

    with open(metrics_file, "a") as f:
        f.write(f"\nMetrics for {file}:\n")
        for metric_name, value in metrics.items():
            if metric_name != "Confusion Matrix":
                f.write(f"{metric_name}: {value:.4f}\n")
        f.write("-" * 50 + "\n")

    pred_overlay = create_overlay(
        orig_img.copy(), preds > 0, alpha=0.25, color=[255, 0, 0]
    )
    truth_overlay = create_overlay(
        orig_img.copy(), label > 0, alpha=0.25, color=[0, 255, 0]
    )

    preds_colored = cv2.cvtColor(preds.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    label_colored = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    combined_img = cv2.hconcat(
        [preds_colored, label_colored, pred_overlay, truth_overlay]
    )
    cv2.imwrite(os.path.join(metrics_output_dir, file), combined_img)

with open(aggregate_metrics_file, "w") as f:
    f.write("Aggregate Metrics (Mean ± Std):\n")
    for metric in ["IoU", "Precision", "Recall", "F1-Score", "Accuracy", "Specificity"]:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")

print("Processing complete. Check the output directory for results.")
