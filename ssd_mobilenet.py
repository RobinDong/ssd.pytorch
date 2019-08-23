import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, cub
from nets import mobilenet
import os


class SSDMobileNetV2(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base MobileNetV2 followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, extras, head, num_classes):
        super(SSDMobileNetV2, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        switcher = {
            2: cub,
            21: voc
        }
        self.cfg = switcher.get(num_classes, coco)
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.backbone = mobilenet.MobileNetV2(num_classes=num_classes, width_mult=1.0)
        self.norm = L2Norm(96, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        _, endpoints = self.backbone(x)
        sources.append(self.norm(endpoints[0]))
        sources.append(endpoints[1])
        x = endpoints[1]

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            #x = v(x)
            #if k % 2 == 1:
            sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


'''def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers'''

def add_extras(cfg):
    layers = []
    block = mobilenet.InvertedResidual

    layers.append(block(320, 512, 2, 512.0/320.0))
    layers.append(block(512, 256, 2, 0.5))
    layers.append(block(256, 256, 2, 1))
    layers.append(block(256, 128, 2, 0.5))

    return layers

def multibox(extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mobilenet_channels = [96, 320]
    for k, channel in enumerate(mobilenet_channels):
        loc_layers += [nn.Conv2d(channel,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channel,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    out_channels = [0, 0, 512, 256, 256, 128]
    for k, v in enumerate(extra_layers, 2):
        loc_layers += [nn.Conv2d(out_channels[k], cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(out_channels[k], cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return extra_layers, (loc_layers, conf_layers)


extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd_mobilenet(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    extras_, head_ = multibox(add_extras(extras[str(size)]),
                              mbox[str(size)], num_classes)
    return SSDMobileNetV2(phase, size, extras_, head_, num_classes)
