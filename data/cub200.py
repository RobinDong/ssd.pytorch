"""Caltech-UCSD Birds 200 Dataset

http://www.vision.caltech.edu/visipedia/CUB-200.html

Original author: Robin Dong
"""
import cv2
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data

from scipy.io import loadmat

CUB_CLASSES = ('bird',)  # Currently, we only use one class 'bird'

class CUBAnnotationTransform(object):
    """Transforms a CUB-200 annotation into a Tensor of bbox coords and label index
    """

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation): the target annotation to be made usable
                will be an MatLab structure
        Returns:
            a list containing lists of bounding boxes [bbox coords, class name]
        """
        res = []

        xmin = target[0][0][0] / width
        ymin = target[1][0][0] / height
        xmax = target[2][0][0] / width
        ymax = target[3][0][0] / height

        # Since we only care about general bird, the label_idx should be constant 0
        res.append([xmin, ymin, xmax, ymax, 0])
        return res


class CUBDetection(data.Dataset):
    """CUB-200 Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to CUB-200 folder.
        image_set (string): imageset to use (eg. 'train', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target 'annotation'
        dataset_name (string, optional): which dataset to load
            (default: 'CUB-200')
    """

    def __init__(self, root, image_set='train', transform=None,
                 target_transform=CUBAnnotationTransform(), dataset_name='CUB-200'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._listpath = osp.join('%s', 'lists', '%s.txt')
        self._annopath = osp.join('%s', 'annotations-mat', '%s.mat')
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self.ids = []
        listfile = self._listpath % (self.root, self.image_set)

        with open(listfile) as fp:
            for line in fp:
                img_id = line.strip().split('.jpg')[0]
                self.ids.append(img_id)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        anno_mat = loadmat(self._annopath % (self.root, img_id))
        target = anno_mat['bbox'][0][0]
        img = cv2.imread(self._imgpath % (self.root, img_id))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # permute HWC to CHW
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == '__main__':
    CUB_ROOT = '/disk3/donghao/data/CUB200/'
    det = CUBDetection(CUB_ROOT)
    with open(CUB_ROOT + 'lists/train.txt') as fp:
        for line in fp:
            line = line.strip().replace('.jpg', '.mat')
            anno_mat = loadmat(CUB_ROOT + 'annotations-mat/{}'.format(line))
            bbox = anno_mat['bbox']

            target_transform = CUBAnnotationTransform()
            res = target_transform(bbox[0][0], 200, 200)
            print(res)
