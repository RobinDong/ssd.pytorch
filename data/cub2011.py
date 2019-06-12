"""Caltech-UCSD Birds-200-2011 Dataset

http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

Original author: Robin Dong
"""
import cv2
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data

CUB_CLASSES = ('bird',)  # Currently, we only use one class 'bird'

class CUBAnnotationTransform(object):
    """Transforms a CUB-200-2011 annotation into a Tensor of bbox coords and label index
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

        xmin = float(target[0]) / width
        ymin = float(target[1]) / height
        xmax = (float(target[0]) + float(target[2])) / width # target[2] is the width of ground truth box
        ymax = (float(target[1]) + float(target[3])) / height # target[3] is the height of ground truth box

        # Since we only care about general bird, the label_idx should be constant 0
        res.append([xmin, ymin, xmax, ymax, 0])
        return res


class CUBDetection(data.Dataset):
    """CUB-200-2011 Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to CUB-200-2011 folder.
        image_set (string): imageset to use (eg. 'train', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target 'annotation'
        dataset_name (string, optional): which dataset to load
            (default: 'CUB-200-2011')
    """

    def __init__(self, root, image_set='train', transform=None,
                 target_transform=CUBAnnotationTransform(), dataset_name='CUB-200-2011'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._listpath = osp.join('%s', 'images.txt')
        self._annopath = osp.join('%s', 'bounding_boxes.txt')
        self._imgpath = osp.join('%s', 'images', '%s.jpg')
        self.ids = []
        listfile = self._listpath % (self.root)
        self.annos = []
        bbfile = self._annopath % (self.root)

        # Load images.txt file
        with open(listfile) as fp:
            for line in fp:
                img_id = line.strip().split()[1].split('.jpg')[0]
                self.ids.append(img_id)
        # Load bounding_boxes.txt file
        with open(bbfile) as fp:
            for line in fp:
                str_bbox = line.strip().split()[1:]
                self.annos.append(str_bbox)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self.annos[index]

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
    CUB_ROOT = '/disk3/donghao/data/CUB_200_2011/'
    det = CUBDetection(CUB_ROOT)
    print(det[1023])
    print(det[3210])

