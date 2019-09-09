import xml.etree.ElementTree as ET
import nvidia.dali.types as types
import nvidia.dali.ops as ops
import os.path as osp
import numpy as np
import collections

from nvidia.dali.pipeline import Pipeline
from data import voc0712, VOC_CLASSES

class VOC0712InputIterator(object):
    def __init__(self, root, batch_size, image_sets=[('2007', 'trainval'), ('2012', 'trainval')]):
        self.root = root
        self.batch_size = batch_size
        self.image_set = image_sets
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __iter__(self):
        self.i = 0
        self.n = len(self.ids)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            img_id = self.ids[self.i]
            target = ET.parse(self._annopath % img_id).getroot()
            fl = open(self._imgpath % img_id, 'rb')

            res = []
            for obj in target.iter('object'):
                difficult = int(obj.find('difficult').text) == 1
                if difficult:
                    continue
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

            batch.append(np.frombuffer(fl.read(), dtype = np.uint8))
            labels.append(np.array(res, dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)
    
    next = __next__


class VOC0712SourcePipeline(Pipeline):
    def __init__(self, iterator, batch_size, num_threads, device_id):
        super(VOC0712SourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.iterator = iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
        self.cast = ops.Cast(device = 'gpu', dtype = types.INT32)

        self.contrast_rng = ops.Uniform(range = (0.5, 1.5))
        self.contrast = ops.Contrast(device = 'gpu')

        self.satu_rng = ops.Uniform(range = (0.5, 1.5))
        self.satu = ops.Saturation(device = 'gpu')

        self.hue_rng = ops.Uniform(range = (-18, 18))
        self.hue = ops.Hue(device = 'gpu')

        self.bright_rng = ops.Uniform(range = (0.8, 1.2))
        self.bright = ops.Brightness(device = 'gpu')

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.contrast(images, contrast = self.contrast_rng())
        images = self.satu(images, saturation = self.satu_rng())
        images = self.hue(images, hue = self.hue_rng())
        images = self.bright(images, brightness = self.bright_rng())
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    vii = VOC0712InputIterator('/home/haodong/data/VOCdevkit/', 1)
    iterator = iter(vii)
    (images, labels) = iterator.next()

    pipe = VOC0712SourcePipeline(iterator, batch_size=1, num_threads=2, device_id = 0)
    pipe.build()

    for _ in range(10):
        pipe_out = pipe.run()

        batch_cpu = pipe_out[0].as_cpu()
        labels_cpu = pipe_out[1]

        label = labels_cpu.at(0)
        print('label:', label)

        img = batch_cpu.at(0)

        plt.imshow(img)
        plt.show()

