import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas
import jpeg4py


class OpenImageAnnotationTransform(object):
    def __call__(self, target, width, height):
        res = []
        for bbox in target:
            label_idx = 0
            final_box = list(bbox)
            final_box.append(label_idx)
            res += [final_box]

        return res


class OpenImageDetection(data.Dataset):
    """ https://storage.googleapis.com/openimages/web/download.html """

    def __init__(self, root, transform=None,
                 target_transform=OpenImageAnnotationTransform()):
        self.img_dir = osp.join(root, 'birds_openimage/')
        self.ids = []
        self.annos = {}
        anno_file = osp.join(root, 'train-annotations-bbox.csv')
        df = pandas.read_csv(anno_file)
        df = df.loc[df['LabelName'] == '/m/015p6']  # Only choose bird

        for index, row in df.iterrows():
            if row['ImageID'] not in self.annos:
                self.annos[row['ImageID']] = []
                self.ids.append(row['ImageID'])
            self.annos[row['ImageID']].append(
                (row['XMin'], row['YMin'], row['XMax'], row['YMax'])
            )

        print('Number of bird-images in OpenImage:', len(self.ids))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, gt, h, w = self.pull_item(index)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self.annos[img_id]

        path = osp.join(self.img_dir, img_id + '.jpg')
        assert osp.exists(path), 'Image path does not exist: {}'. format(path)

        #img = cv2.imread(path)
        img = jpeg4py.JPEG(path).decode()
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == '__main__':
    detect = OpenImageDetection('/home/haodong/data/')
    index = 9999
    img, boxes = detect[index]
    print(boxes)
    img = cv2.imread('/home/haodong/data/birds_openimage/'
                     + detect.ids[index] + '.jpg')
    img = img.astype('uint8')
    height, width, _ = img.shape

    for index in range(len(boxes)):
        xmin = int(boxes[index][0] * width)
        ymin = int(boxes[index][1] * height)
        xmax = int(boxes[index][2] * width)
        ymax = int(boxes[index][3] * height)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)

    cv2.imwrite('opendraw.jpg', img)
