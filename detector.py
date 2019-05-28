import cv2
import time
import torch

from torch.autograd import Variable
from ssd import build_ssd
from data import BaseTransform

def main():
    # Load net
    num_classes = 21
    net = build_ssd('test', 300, num_classes)
    ckpt = torch.load('weights/ssd300_COCO_120000.pth', map_location = 'cpu')
    net.load_state_dict(ckpt)
    net.eval()
    print('Finished loading model!')

    begin = time.time()
    transform = BaseTransform(net.size, (104, 117, 123))
    img = cv2.imread('WechatIMG17.jpeg')
    #img = cv2.imread('bird_matrix.jpg')
    height, width = img.shape[:2]
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    detections = net(x).data

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    scale = torch.Tensor([width, height, width, height])
    bird_index = 3
    j = 0
    while detections[0, bird_index, j, 0] >= 0.5:
        pt = (detections[0, bird_index, j, 1:] * scale).cpu().numpy()
        score = detections[0, bird_index, j, 0]
        print('pt', pt, score)
        cv2.rectangle(img,
                      (int(pt[0]), int(pt[1])),
                      (int(pt[2]), int(pt[3])),
                      COLORS[bird_index % 3], 2)
        cv2.putText(img, 'bird_'+str(score.item()), (int(pt[0]), int(pt[1])),
                    FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
        j += 1
    print('time', time.time() - begin)
    print('objects', j)

    cv2.imwrite('draw.jpg', img)

main()
