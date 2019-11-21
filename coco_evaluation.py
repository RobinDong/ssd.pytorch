import cv2
import json
import time
import torch
import argparse
import os.path as osp

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable, Function
from data import *
from bn_fusion import fuse_bn_recursively

ANNOTATIONS = 'annotations'
INSTANCES_SET = 'instances_{}.json'
ROOT = '/home/haodong/data/coco/'

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, variance, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = variance

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


def detect_one_image(net, transform, detect, img_id, img_path):
    #img = cv2.imread('17test.png')
    #img = cv2.imread('WechatIMG17.jpeg')
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    result = net(x)
    softmax = torch.nn.Softmax(dim=-1)
    loc = result[0]
    conf = result[1]
    priors = result[2]
    detections = detect(loc, softmax(conf), priors).data

    scale = torch.Tensor([width, height, width, height])
    j = 0
    results = []
    bird_index = 1
    score = detections[0, bird_index, j, 0]
    while score >= 0.5:
        pt = (detections[0, bird_index, j, 1:] * scale).cpu().numpy()
        print('pt', pt.tolist(), score.item())
        pt = pt.tolist()
        results.append({
            'image_id': img_id,
            'category_id': 16,
            'bbox': [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1]],
            'score': score.item()
        })
        j += 1
        score = detections[0, bird_index, j, 0]
    return results

def detection(args, coco):
    # Load net
    bird_index = 1
    if args.dataset == 'VOC':
        cfg = voc
    elif args.dataset == 'CUB':
        cfg = cub

    net = torch.load('weights/ssd300_COCO_{}.pth'.format(args.trained_model), map_location = 'cpu')
    net = fuse_bn_recursively(net)
    net.eval()
    print('Finished loading model!')

    transform = BaseTransform(net.size, (104, 117, 123))
    detect = Detect(cfg['num_classes'], cfg['variance'], bkg_label=0, top_k=200,
                    conf_thresh=0.01, nms_thresh=0.45)

    img_ids = []
    for img_id in list(coco.imgToAnns.keys()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        nr_birds = 0
        for obj in target:
            if obj['category_id'] == 16: # Only use images contains bird
                nr_birds += 1
        if nr_birds > 0:
            img_ids.append(img_id)

    print('Processing {} images...'.format(len(img_ids)))
    results = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_path = osp.join(ROOT, 'images', '{}'.format(args.year),
                            coco.loadImgs(img_id)[0]['file_name'])
        results.extend(detect_one_image(net, transform, detect, img_id, img_path))

    with open('coco_bird_pred_{}.json'.format(args.year), 'w') as fp:
        json.dump(results, fp)


def main(args):
    coco = COCO(osp.join(ROOT, ANNOTATIONS, INSTANCES_SET.format('{}'.format(args.year))))

    if not osp.exists('coco_bird_pred_{}.json'.format(args.year)):
        detection(args, coco)

    coco_src = coco.loadRes('coco_bird_eval_{}.json'.format(args.year))
    coco_target = coco.loadRes('coco_bird_pred_{}.json'.format(args.year))
    coco_eval = COCOeval(coco_src, coco_target)
    coco_eval.params.useSegm = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'CUB'],
                        type=str, help='VOC or CUB')
    parser.add_argument('--trained_model', default=200000,
                        type=int, help='trained model number for predicting')
    parser.add_argument('--year', default='val2014',
                        type=str, help='dataset used to be source')
    args = parser.parse_args()
    main(args)
