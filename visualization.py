import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import config
import numpy as np

S, B, C = config.S, config.B, config.C


def _draw_single_box(ax, cx, cy, w, h, label, color, linestyle='-', text=None):
    """Draws a single bounding box with label text."""
    top_left_x = cx - w / 2
    top_left_y = cy - h / 2

    rect = patches.Rectangle(
        (top_left_x, top_left_y), w, h,
        linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle
    )
    ax.add_patch(rect)

    if text:
        ax.text(top_left_x, top_left_y - 2, text,
                color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))


def _process_and_draw_boxes(ax, target, class_names, image_size, conf_thresh=0.0, is_gt=False):
    """Processes YOLO grid and draws boxes on the image."""
    grid_size_x = image_size[0] / S
    grid_size_y = image_size[1] / S

    for row in range(S):
        for col in range(S):
            cell = target[row, col]
            class_probs = cell[:C]
            bbox_data = cell[C:]

            if class_probs.sum() == 0:
                continue

            class_probs = torch.softmax(class_probs, dim=0)
            class_idx = torch.argmax(class_probs).item()
            label = class_names[class_idx]
            class_prob = class_probs[class_idx].item()

            for b in range(B):
                start = b * 5
                conf = bbox_data[start + 4]
                if conf < conf_thresh:
                    continue

                rel_x, rel_y, w, h, _ = bbox_data[start:start + 5]
                cx = col * grid_size_x + rel_x * image_size[0]
                cy = row * grid_size_y + rel_y * image_size[1]
                abs_w = w * image_size[0]
                abs_h = h * image_size[1]

                # Skip boxes with centers lying outside image dimensions
                if not (0 <= rel_x <= 1 and 0 <= rel_y <= 1):
                    continue
                if not (0 <= w <= 1 and 0 <= h <= 1):
                    continue


                color = 'g' if is_gt else 'r'
                linestyle = '-'
                prefix = 'GT' if is_gt else 'Pred'

                if is_gt:
                    text = f'{prefix}: {label}'
                else:
                    text = f'{prefix}: {label} ({class_prob:.2f})'

                _draw_single_box(ax, cx, cy, abs_w, abs_h, label, color, linestyle, text)


def visualize_gt(data, target, class_names=list(config.classes.keys())):
    """Visualizes GT bounding boxes."""
    image = data.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)  
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    _process_and_draw_boxes(ax, target, class_names, config.IMAGE_SIZE, conf_thresh=0.0, is_gt=True)
    plt.axis('off')
    plt.show()


def visualize_pred(data, target, class_names=list(config.classes.keys()), conf_thresh=0.1):
    """Visualizes predicted bounding boxes with class prob."""
    image = data.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)  
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    _process_and_draw_boxes(ax, target, class_names, config.IMAGE_SIZE, conf_thresh=conf_thresh, is_gt=False)
    plt.axis('off')
    plt.show()


def visualize_pred_with_gt(data, pred, gt, class_names=list(config.classes.keys()), conf_thresh=0.4):
    """Visualizes both predicted and ground truth bounding boxes."""
    image = data.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1) 
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    _process_and_draw_boxes(ax, gt, class_names, config.IMAGE_SIZE, conf_thresh=0.0, is_gt=True)
    _process_and_draw_boxes(ax, pred, class_names, config.IMAGE_SIZE, conf_thresh=conf_thresh, is_gt=False)
    plt.axis('off')
    plt.show()
