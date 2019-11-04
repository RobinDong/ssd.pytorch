# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/home/haodong/ssd.pytorch/")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 2,
    'lr_steps': (9000, 40000),
    'max_iter': 40001,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 30, 60, 100, 150, 300],
    'min_sizes': [45, 90, 135, 180, 225, 270],
    'max_sizes': [90, 135, 180, 225, 270, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

cub = {
    'num_classes': 2,
    'lr_steps': (9000, 40000),
    'max_iter': 40001,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 30, 60, 100, 150, 300],
    'min_sizes': [45, 90, 135, 180, 225, 270],
    'max_sizes': [90, 135, 180, 225, 270, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CUB',
}

coco = {
    'num_classes': 2,
    'lr_steps': (9000, 400000),
    'max_iter': 400001,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 30, 60, 100, 150, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
