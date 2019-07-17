import numpy as np
import cv2

import xml.etree.ElementTree as ET


class Background(object):
    def __init__(self, root_dir = '/home/hdong/data/VOCdevkit/VOC2012/',
                 limit=1000, exclude_class='bird', with_foreground=False):
        list_file = '{}/ImageSets/Main/{}_trainval.txt'.format(root_dir, exclude_class)

        self.image_list = []
        for line in open(list_file):
            name, tag = line.strip().split()
            if tag == '-1': # The image without tag of 'exclude_class'
                # Read image
                image_file = root_dir + 'JPEGImages/' + name + '.jpg'
                img = cv2.imread(image_file)
                if img is None:
                    print('Failed to read {}'.format(image_file))
                    continue
                # Read annotations
                annotation_file = root_dir + 'Annotations/' + name + '.xml'
                raw_target = ET.parse(annotation_file).getroot()
                target = self.target_transform(raw_target)

                total_area = 0 # Not choose images with less than 0.5 foreground
                if not with_foreground: # Crop out all objects
                    for obj in target:
                        xmin = obj[0]
                        ymin = obj[1]
                        xmax = obj[2]
                        ymax = obj[3]
                        img[ymin:ymax, xmin:xmax, :] = 0
                        total_area += (xmax - xmin) * (ymax - ymin)

                height, width, depth = img.shape
                if with_foreground or ((total_area * 2) < (height * width)):
                    self.image_list.append((img, target))
                if len(self.image_list) > limit:
                    break

    def __getitem__(self, index):
        return self.image_list[index]

    def __len__(self):
        return len(self.image_list)

    def target_transform(self, raw_target):
        res = []
        for obj in raw_target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            res += [bndbox]  # [xmin, ymin, xmax, ymax]
        return np.array(res)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    bg = Background()

    for index in range(10):
        img, boxes = bg[index]

        for index in range(len(boxes)):
            xmin = boxes[index][0]
            ymin = boxes[index][1]
            xmax = boxes[index][2]
            ymax = boxes[index][3]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

        img = cv2.resize(img, (500, 500))
        plt.imshow(img)
        plt.show()
