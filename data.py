import config
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import utils
import random
import os
from glob import glob
from PIL import Image
import xml.etree.ElementTree as ET

class VOC_to_YOLO_Dataset(Dataset):
    def __init__(self, use_manual=False, set_type='trainval', normalize=False, augment=False):
        self.use_manual = use_manual
        self.augment = augment
        self.normalize = normalize
        self.classes = config.classes

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])

        if not os.path.exists(os.path.join(config.DATA_PATH, "VOCdevkit")):
            _ = VOCDetection(
                root=config.DATA_PATH,
                year='2012',
                image_set='trainval',
                download=True
            )

        if use_manual:
            self.image_dir = os.path.join(config.DATA_PATH, "VOCdevkit/VOC2012/JPEGImages")
            self.ann_dir = os.path.join(config.DATA_PATH, "VOCdevkit/VOC2012/Annotations")

            self.image_paths = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
            self.ann_paths = [os.path.join(self.ann_dir, os.path.basename(p).replace(".jpg", ".xml")) for p in self.image_paths]
            self.valid_pairs = [(img, ann) for img, ann in zip(self.image_paths, self.ann_paths) if os.path.exists(ann)]
        else:
            assert set_type in {'trainval', 'train', 'test'}
            self.dataset = VOCDetection(
                root=config.DATA_PATH,
                year='2012',
                image_set=set_type,
                download=True,
                transform=self.transform
            )

    def __len__(self):
        return len(self.valid_pairs) if self.use_manual else len(self.dataset)

    def __getitem__(self, i):
        if self.use_manual:
            image_path, ann_path = self.valid_pairs[i]
            image = Image.open(image_path).convert("RGB")
            data = self.transform(image)
            label = self.parse_annotation(ann_path)
        else:
            data, label = self.dataset[i]

        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        size = label['annotation']['size']
        width, height = int(size['width']), int(size['height'])
        x_scale = config.IMAGE_SIZE[0] / width
        y_scale = config.IMAGE_SIZE[1] / height

        boxes = []
        objects = label['annotation']['object']
        if isinstance(objects, dict):  # If only one object, VOC format gives dict not list
            objects = [objects]
        for obj in objects:
            box = obj['bndbox']
            coords = (
                int(float(box['xmin']) * x_scale),
                int(float(box['xmax']) * x_scale),
                int(float(box['ymin']) * y_scale),
                int(float(box['ymax']) * y_scale)
            )
            name = obj['name']
            boxes.append((name, coords))

        grid_size_x = data.size(dim=2) / config.S
        grid_size_y = data.size(dim=1) / config.S

        ground_truth = torch.zeros((config.S, config.S, 5 * config.B + config.C))
        bbox_counter = {}
        class_names = {}

        for name, (x_min, x_max, y_min, y_max) in boxes:
            assert name in self.classes
            class_index = self.classes[name]

            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    bbox_index = bbox_counter.get(cell, 0)
                    if bbox_index < config.B:
                        bbox = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],
                            (x_max - x_min) / config.IMAGE_SIZE[0],
                            (y_max - y_min) / config.IMAGE_SIZE[1],
                            1.0
                        )
                        start = 5 * bbox_index + config.C
                        ground_truth[row, col, start:] = torch.tensor(bbox).repeat(config.B - bbox_index)
                        bbox_counter[cell] = bbox_index + 1

        return data, ground_truth

    def parse_annotation(self, ann_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(ann_path)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            box = {
                'xmin': int(float(bndbox.find("xmin").text)),
                'ymin': int(float(bndbox.find("ymin").text)),
                'xmax': int(float(bndbox.find("xmax").text)),
                'ymax': int(float(bndbox.find("ymax").text))
            }
            objects.append({'name': name, 'bndbox': box})

        return {'annotation': {'size': {'width': width, 'height': height}, 'object': objects}}
