import imp
import re
import sys
from cv2 import threshold

from torch import rand, zero_
from mmseg.apis import inference_segmentor, init_segmentor
import argparse
import os
import numpy as np

import shutil



parser = argparse.ArgumentParser()
parser.add_argument(
    '-img_dir','-i', required=False, type=str, help='Directory path of infer dataset')
args = parser.parse_args()

img_dir = args.img_dir
img_dir = './ade20k/images/'
out_img_dir = './ade20k/result/'

# config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
# checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py'
checkpoint_file = 'checkpoints/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'

CLASSES = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag')

img_list = [
    c for c in os.listdir(img_dir)
    if not os.path.isdir(os.path.join(img_dir, c))
]
# print(img_list)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
count_num = 0
threshold_cp = 0.2

for iter in img_list:
    img = os.path.join(img_dir, iter)
    result = inference_segmentor(model, img)

    output = np.array(result)
    # print(output.shape)
    # print(output.size)

    r_map = {2:0, 4:0, 6:0}
    for c in output:
        for h in c:
            for w in h:
                if w in r_map:
                    r_map[w] += 1
                else:
                    r_map[w] = 1

    # for key in r_map:
    #     print(str(key) + ':' + str(r_map[key]), end=' ')
    # print()
    count = output.shape[1] * output.shape[2]
    try:
        sky_rate = r_map[2] * 1.0 / count
        # print('sky_rate is {}'.format(sky_rate))
        road_rate = r_map[6] * 1.0 / count
        # print('road_rate is {}'.format(road_rate))
        tree_rate = r_map[4] * 1.0 / count
        # print('tree_rate is {}'.format(tree_rate))
        if not (road_rate > threshold_cp or sky_rate > threshold_cp or tree_rate > threshold_cp):
            continue
        # print("img name: {}".format(img))
        new_name = out_img_dir + img[img.rfind('/')+1 :]
        shutil.copyfile(img, new_name)

        count_num += 1
        print('deal img :'.format(count_num), end=' ')

    except Exception as e:
        print("exception: {}".format(e))
        continue
    # model.show_result(img, result, out_file= out_img_dir + img[img.rfind('/')+1:], opacity=0.5)


print("finished deal {}".format(count_num))