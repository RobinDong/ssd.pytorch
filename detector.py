import cv2
import time
import torch
import argparse

from torch.autograd import Variable
from ssd import build_ssd
from ssd_resnext import build_ssd_resnext
from ssd_mobilenet import build_ssd_mobilenet
from data import *

def main(args):
    # Load net
    if args.dataset == 'VOC':
        cfg = voc
        bird_index = 3
    elif args.dataset == 'CUB':
        cfg = cub
        bird_index = 1

    if args.backbone == 'vgg':
        net = build_ssd('test', 300, cfg['num_classes'])
    elif args.backbone == 'resnext':
        net = build_ssd_resnext('test', 300, cfg['num_classes'])
    elif args.backbone == 'mobilenet':
        net = build_ssd_mobilenet('test', 300, cfg['num_classes'])
    ckpt = torch.load('weights/ssd300_COCO_{}.pth'.format(args.trained_model), map_location = 'cpu')
    net.load_state_dict(ckpt)
    net.eval()
    print('Finished loading model!')

    begin = time.time()
    transform = BaseTransform(net.size, (104, 117, 123))
    #img = cv2.imread('17test.png')
    #img = cv2.imread('WechatIMG17.jpeg')
    img = cv2.imread('bird_matrix.jpg')
    height, width = img.shape[:2]
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    detections = net(x).data

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    scale = torch.Tensor([width, height, width, height])
    j = 0
    score = detections[0, bird_index, j, 0]
    while score >= 0.5:
        pt = (detections[0, bird_index, j, 1:] * scale).cpu().numpy()
        print('pt', pt, score)
        cv2.rectangle(img,
                      (int(pt[0]), int(pt[1])),
                      (int(pt[2]), int(pt[3])),
                      COLORS[bird_index % 3], 2)
        cv2.putText(img, 'bird_{:.2f}'.format(score.item()), (int(pt[0]), int(pt[1])),
                    FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
        j += 1
        score = detections[0, bird_index, j, 0]
    print('time', time.time() - begin)
    print('objects', j)

    cv2.imwrite('draw.jpg', img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'CUB'],
                        type=str, help='VOC or CUB')
    parser.add_argument('--trained_model', default=120000,
                        type=int, help='trained model number for predicting')
    parser.add_argument('--backbone', default='vgg', choices=['vgg', 'resnext', 'mobilenet'],
                        type=str, help='Backbone network')
    args = parser.parse_args()
    main(args)
