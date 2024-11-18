from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np
from tqdm import tqdm
from .utils import get_all_combinations, get_weight_map
from torchmetrics import ConfusionMatrix
import logging
import os
import sys
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def visualize(model, layer, image, use_cuda=False):
    cam = GradCAM(model=model, target_layer=layer, use_cuda=use_cuda)
    return cam
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam(input_tensor=image.unsqueeze(0))[0, :]), cv2.COLORMAP_JET
    )
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + image.permute(1, 2, 0).numpy()
    cam = cam / np.max(cam)
    return cam


def train_all_epochs(
    model,
    train_loader,
    valid_loader,
    test_loader,
    epochs,
    criterions,
    metrics,
    monitor_metric,
    writer,
    optimizer,
    accumulate_batch,
    criterion_scaling=None,
    average_outputs=False,
    name: str = "",
    weight_map=False,
    save_folder="",
    early_stopping=10,
    modelstring="",
    custom_data={},
    distance_map_loss=None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=False if device == "cpu" else True)
    model = model.to(device)
    has_distance_map = True if distance_map_loss is not None else False
    accumulate_every = accumulate_batch // train_loader.batch_size
    best_metric = float("inf")
    best_dict = {}
    cnt = 0
    for epoch in range(epochs):
        training_bar = tqdm(train_loader, file=sys.stdout)
        train_dict = train_one_epoch(
            model,
            training_bar,
            criterions,
            criterion_scaling,
            average_outputs,
            device,
            epoch,
            optimizer,
            scaler,
            metrics,
            writer,
            name,
            accumulate_every,
            best_metric,
            weight_map,
            distance_map_loss,
        )
        if valid_loader:
            valid_bar = tqdm(valid_loader, file=sys.stdout)
            tmp, m = evaluate(
                model,
                valid_bar,
                criterions=criterions,
                criterion_scaling=criterion_scaling,
                writer=writer,
                metrics=metrics,
                monitor_metric=monitor_metric,
                device=device,
                epoch=epoch,
                name=name,
                average_outputs=False,
                distance_map_loss=distance_map_loss,
            )
            if best_metric >= tmp:
                best_metric = tmp
                best_dict = m
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model": modelstring,
                        "has_distance_map": has_distance_map,
                        "validation_metrics": best_dict,
                        "train_metrics": train_dict,
                        "custom_data": custom_data,
                    },
                    os.path.join(save_folder, name + ".pt"),
                )
                cnt = 0
            else:
                cnt += 1
            if cnt >= early_stopping:
                model.load_state_dict(
                    torch.load(os.path.join(save_folder, name + ".pt"))[
                        "model_state_dict"
                    ]
                )
                return


def get_distance_map(mask):
    mask = mask.clone()
    mask = mask.type(torch.uint8)
    mask = mask.numpy()
    mask[mask > 0] = 255
    distances = []
    for cnt, ma in enumerate(mask):
        dist = cv2.distanceTransform(ma, cv2.DIST_L2, 5)
        distances.append(cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX))
        dist[dist <= 0.7] = 0
        # cv2.imwrite(f"{cnt}.png", dist * 255.0)
        # cv2.imwrite(f"{cnt}a.png", ma)
    a = np.stack(np.array(distances), axis=0)
    return torch.tensor(a, dtype=torch.float)


def get_distance_map_fixed(mask):
    mask = mask.clone()
    mask = mask.type(torch.uint8)
    mask = mask.numpy()
    mask[mask > 0] = 255  # set all classes to the same value
    distances = []
    for cnt2, img in enumerate(mask):
        to = np.zeros_like(img, dtype=np.float32)
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        for i, cnt in enumerate(contours):
            mask2 = np.zeros_like(img)
            cv2.drawContours(mask2, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
            ab = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            pts = np.where(ab > 0)
            to[pts[0], pts[1]] = ab[pts[0], pts[1]]
        # print(cv2.imwrite(f"{cnt2}.png", to*255.))
        # print(cv2.imwrite(f"{cnt2}a.png", img))
        # print(cv2.imwrite("hier.png", img))

        distances.append(to)
    # return distances
    a = np.stack(np.array(distances), axis=0)
    return torch.tensor(a, dtype=torch.float)


def get_skeleton(mask):
    # mask[mask > 0] = 255 #set all classes to the same value
    mask = mask.clone()
    mask = mask.type(torch.uint8)
    mask = mask.numpy()
    mask[mask > 0] = 255  # set all classes to the same value
    distances = []
    for cnt2, img in enumerate(mask):
        to = np.zeros_like(img, dtype=np.float32)
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        for i, cnt in enumerate(contours):
            mask2 = np.zeros_like(img)
            cv2.drawContours(mask2, [cnt], -1, 1, -1)
            skeleton = skeletonize(mask2)
            dist = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
            ab = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            pts = np.where(skeleton > 0)
            to[pts[0], pts[1]] = ab[pts[0], pts[1]]
        distances.append(to)

    a = np.stack(np.array(distances), axis=0)
    return torch.tensor(a, dtype=torch.float)


def train_one_epoch(
    model,
    training_bar,
    criterions,
    criterion_scaling,
    average_outputs=False,
    device=None,
    epoch=0,
    optimizer=None,
    scaler=None,
    metrics=None,
    writer=None,
    name: str = "",
    accumulate_every=1,
    best_metric=0.0,
    weight_map=False,
    distance_map_loss=None,
):
    sig = torch.nn.Sigmoid()
    for metric in metrics:
        metric.reset()
    total_loss = 0.0
    train_dict = {}
    for cnt, (x, y) in enumerate(training_bar):
        x = x.to(device)
        if distance_map_loss:
            dist = get_distance_map_fixed(y)
            dist = dist.to(device)
        y = y.to(device)
        # TODO Implement multiple outputs of network
        with torch.cuda.amp.autocast():
            loss = None
            try:
                predictions = model(x)
            except:
                continue
            if distance_map_loss:
                distance_map_predictions = sig(predictions[:, -1, :, :])
                predictions = predictions[:, 0:-1, :, :]
            weight_maps = (
                get_weight_map(y.detach().cpu().numpy() * 255.0).to(device)
                if weight_map
                else None
            )
            loss = criterions(predictions, y, weight_maps)
            if distance_map_loss:
                loss += distance_map_loss(distance_map_predictions, dist)
            predictions = torch.argmax(predictions, 1)
        for metric in metrics:
            metric.update(predictions.detach().cpu(), y.detach().cpu())
        scaler.scale(loss).backward()
        # gradient accumulation
        if (cnt > 0 and cnt % accumulate_every == 0) or cnt == len(training_bar) - 1:
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        metric_str = "Train: Epoch:%i, Loss:%.4f, Best:%.4f " + "".join(
            [metric.__class__.__name__ + ":%.4f, " for metric in metrics]
        )
        epoch_values = [metric.compute().item() for metric in metrics]

        for metric, val in zip(metrics, epoch_values):
            writer.add_scalar(
                f"train/train-{name}-{metric.__class__.__name__}", val, epoch
            )

        total_loss += loss.item()
        current_loss = total_loss / float((cnt + 1))
        writer.add_scalar(f"train/train-loss", current_loss, epoch)
        training_bar.set_description(
            metric_str % tuple([epoch + 1, current_loss, best_metric] + epoch_values)
        )

    epoch_values = [metric.compute().item() for metric in metrics]
    for metric, val in zip(metrics, epoch_values):
        train_dict[metric.__class__.__name__] = val
    for metric in metrics:
        metric.reset()
    train_dict["train_loss"] = total_loss / len(training_bar)
    return train_dict


def evaluate(
    model,
    valid_bar,
    criterions,
    criterion_scaling,
    writer,
    metrics,
    monitor_metric,
    device,
    epoch,
    name,
    average_outputs=False,
    distance_map_loss=None,
):
    assert writer is not None
    assert metrics is not None
    assert monitor_metric is not None
    best_dict = {}
    monitor_metric.reset()
    for metric in metrics:
        metric.reset()
    model.eval()
    total_loss = 0.0
    for cnt, (x, y) in enumerate(valid_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        # TODO implement average_outputs
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                try:
                    predictions = model(x)
                    if distance_map_loss:
                        predictions = predictions[:, 0:-1, :, :]
                except:
                    continue
            loss = criterions(predictions, y)
            predictions = torch.argmax(predictions, 1)
        monitor_metric.update(predictions.detach().cpu(), y.detach().cpu())
        for metric in metrics:
            metric.update(predictions.detach().cpu(), y.detach().cpu())
        metric_str = "Valid: Epoch:%i, Loss:%.4f, " + "".join(
            [metric.__class__.__name__ + ":%.4f, " for metric in metrics]
        )
        epoch_values = [metric.compute().item() for metric in metrics]

        for metric, val in zip(metrics, epoch_values):
            writer.add_scalar(
                f"valid/valid-{name}-{metric.__class__.__name__}", val, epoch
            )

        total_loss += loss.item()
        current_loss = total_loss / float((cnt + 1))
        writer.add_scalar(f"valid/valid-loss", current_loss, epoch)
        valid_bar.set_description(
            metric_str % tuple([epoch + 1, current_loss] + epoch_values)
        )
    epoch_values = [metric.compute().item() for metric in metrics]
    for metric, val in zip(metrics, epoch_values):
        best_dict[metric.__class__.__name__] = val
    for metric in metrics:
        metric.reset()
    model.zero_grad()
    model.train()

    best_dict["validation_monitor_metric"] = monitor_metric.compute().item()
    best_dict["validation_loss"] = total_loss / len(valid_bar)
    return total_loss / len(valid_bar), best_dict


def test_trainer(models: list, test_loaders, metrics, distance_map_loss=None):
    assert test_loaders
    assert metrics
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_dict = {}
    combinations = get_all_combinations(
        models
    )  # get all permutations of the model list
    for cnt, model_comb in enumerate(tqdm(combinations, file=sys.stdout)):
        for metric in metrics:
            metric.reset()
        names = ",".join([x.name for x in model_comb])
        for x, y in test_loaders[0]:
            predictions = None
            x = x.to(device)
            y = y.to(device)
            for model in model_comb:
                model.model.eval()
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if predictions is None:
                            predictions = model(x).detach().cpu()
                        else:
                            predictions += model(x).detach().cpu()
            predictions = torch.argmax(predictions, 1)
            for metric in metrics:
                metric.update(predictions.detach().cpu(), y.detach().cpu())
        log_dict[names] = [metric.compute().item() for metric in metrics]
    return log_dict


def make_single_class_per_contour(img, min_size=None):
    img = img.astype(np.uint8)
    orig = img.copy()
    img[img > 0] = 255
    contours, hierarchy = cv2.findContours(
        image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        mask = np.zeros_like(orig)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        pts = np.where(mask == 255)
        vals = np.unique(orig[pts[0], pts[1]], return_counts=True)
        orig[pts[0], pts[1]] = vals[0][np.argmax(vals[1])]
        if min_size is not None:
            if area < min_size:
                orig[pts[0], pts[1]] = 0
    return orig


def predict_instance_segmentation(model, images, device, confidence=0.35):
    model.eval()
    model = model.to(device)
    all_predictions = []
    for cnt, image in enumerate(images):
        image = image / 255.0
        image = torch.unsqueeze(
            torch.tensor(image, dtype=torch.float).permute(2, 0, 1), 0
        )
        image = image.to(device)
        model.zero_grad()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                predictions = model(image)

        u = len(
            list(filter(lambda x: x >= confidence, list(predictions[0].values())[2]))
        )
        all_predictions.append(
            [
                list(predictions[0].values())[i].cpu().detach().numpy()[0:u]
                for i in range(4)
            ]
        )
    return all_predictions


def predict_classification_images(model, images, device):
    model.eval()
    model = model.to(device)
    all_predictions = []
    for cnt, image in enumerate(images):
        image = image / 255.0
        image = torch.unsqueeze(
            torch.tensor(image, dtype=torch.float).permute(2, 0, 1), 0
        )
        image = image.to(device)
        model.zero_grad()
        # TODO implement average_outputs
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                predictions = model(image)

            predictions = predictions[0].detach().cpu().numpy()
            all_predictions.append(predictions)
    return all_predictions


def predict_images(
    model,
    images,
    device,
    single_class_per_contour=False,
    min_size=None,
    has_distance_map=False,
    fill_holes=False,
):
    model.eval()
    total_loss = 0.0
    model = model.to(device)
    all_predictions = []
    all_distance_maps = []
    sig = torch.nn.Sigmoid()
    for cnt, image in enumerate(images):
        image = image / 255.0
        image = torch.unsqueeze(
            torch.tensor(image, dtype=torch.float).permute(2, 0, 1), 0
        )
        image = image.to(device)
        model.zero_grad()
        # TODO implement average_outputs
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                predictions = model(image)
                if has_distance_map:
                    distance_map = predictions[
                        :, -1
                    ]  # if trained with distance map the last output class corresponds to the distance map which is not normalized with sigmoid at this point
                    predictions = predictions[:, 0:-1]  # classes [0:n-2]
            predictions = torch.argmax(predictions, 1)

            predictions = predictions[0].detach().cpu().numpy()
            if single_class_per_contour:
                predictions = make_single_class_per_contour(predictions, min_size)
            if fill_holes:
                unique_values = np.unique(predictions)
                all_classes = np.zeros_like(predictions)
                for val in unique_values:
                    if val == 0:
                        continue
                    tmp = np.zeros_like(predictions)
                    tmp[predictions == val] = 1
                    tmp = np.uint8(ndi.binary_fill_holes(tmp))
                    tmp = tmp.astype(np.int32)
                    all_classes[tmp == 1] = val
                predictions = all_classes
            all_predictions.append(predictions)
            if has_distance_map:
                distance_map = sig(distance_map)
                all_distance_maps.append(distance_map[0].detach().cpu().numpy())

    return all_predictions, all_distance_maps
