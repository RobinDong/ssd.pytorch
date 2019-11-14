import json
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNOTATIONS = 'annotations'
INSTANCES_SET = 'instances_{}.json'

dataset = 'val2017'
coco = COCO(osp.join('/home/haodong/data/coco/', ANNOTATIONS,
                     INSTANCES_SET.format(dataset)))

if not osp.exists('coco_bird_eval_{}.json'.format(dataset)):
    results = []

    for img_id in list(coco.imgToAnns.keys()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        for obj in target:
            if obj['category_id'] == 16: # Only use images contains bird
                results.append({
                    'image_id': img_id,
                    'category_id': 16,
                    'bbox': obj['bbox'],
                    'score': 1.0
                })

    with open('coco_bird_eval_{}.json'.format(dataset), 'w') as fp:
        json.dump(results, fp)
