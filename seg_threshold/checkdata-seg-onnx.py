from cv2 import cvtColor
import onnx
import numpy as np
import onnxruntime
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    '-model_path','-m', required=True, type=str, help='Directory path of onnx model')
parser.add_argument(
    '-model_type','-t', required=False, type=bool, default='', help='type of onnx model is mmcv')

parser.add_argument(
    '-img_dir','-i', required=False, type=str, help='Directory path of infer dataset')
args = parser.parse_args()

onnx_file = args.model_path
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print('The model is checked!')


# means = [0.5, 0.5, 0.5]
# stds = [0.5, 0.5, 0.5]
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

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

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


def cv2_transform(cv2_img):
    img = cv2_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(cv2_img, (512, 512), Image.BILINEAR)
    # img = cv2.resize(cv2_img, (224, 224), Image.BILINEAR)
    img = np.array(img[:, :, ::-1], dtype=np.float32)
    img = img / 255
    img = img - np.array(means, dtype=np.float32)
    img = img / np.array(stds, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]
    img = torch.from_numpy(img)
    return img


def show_result(
                img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                classes=CLASSES,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img =  cv2.imread(img)
    img_origin = img
    img = cv2.resize(img, (512, 512), Image.BILINEAR)
    img = img.copy()
    seg = result[0]
    if palette is None:
        # Get random state before set seed,
        # and restore random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(len(classes), 3))
        np.random.set_state(state)
    palette = np.array(palette)
    assert palette.shape[0] == len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    img = cv2.resize(img, (img_origin.shape[1], img_origin.shape[0]), Image.BILINEAR)
    if show:
        cv2.imshow(win_name, img)
    if out_file is not None:
        cv2.imwrite(out_file, img)

    if not (show or out_file):
        print('show==False and out_file is not specified, only '
                        'result image will be returned')
    return img

resnet_session = onnxruntime.InferenceSession(onnx_file)

img_dir = args.img_dir
# img_dir = './ade20k/images/'
# out_img_dir = './ade20k/result/'
out_img_dir = os.path.join(img_dir, '../result/')
render_img_dir = os.path.join(img_dir, '../result/render/')
if not os.path.exists(out_img_dir):
    os.makedirs(out_img_dir)
if not os.path.exists(render_img_dir):
    os.makedirs(render_img_dir)

img_list = [
    c for c in os.listdir(img_dir)
    if not os.path.isdir(os.path.join(img_dir, c))
]


threshold_cp = 0
count_num = 0
is_render = True
is_mmcv = args.model_type
is_show_result = True

for iter in img_list:
    img_path = os.path.join(img_dir, iter)

    # image = Image.open(img_path)
    # img = get_test_transform()(image)
    # img = img.unsqueeze_(0)
    # # print(img.shape)

    img = cv2.imread(img_path)
    img = cv2_transform(img)
    # print("input imageop mean {} and std {}".format(img.mean(), img.std()))
    # print(imageop.shape)


    inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
    outs = resnet_session.run(None, inputs)[0]

    # print("onnx result {}  shape {}".format(
    #     outs, outs.shape))

    output = np.array(outs)
    print(output.shape)
    # print(output.size)

    if is_mmcv:
        output = output[0]
        outs = outs[0]
    count = output.shape[1] * output.shape[2]

    r_map = {2:0, 4:0, 6:0}
    for c in output:
        for h in c:
            for w in h:
                if w in r_map:
                    r_map[w] += 1
                else:
                    r_map[w] = 1
    if is_show_result:
        for key in sorted(r_map):
            print(str(key) + ':' + str(r_map[key]), end=' ')
        print()

    try:
        sky_rate = r_map[2] * 1.0 / count
        road_rate = r_map[6] * 1.0 / count
        tree_rate = r_map[4] * 1.0 / count
        if is_show_result:
            print('sky_rate is {}'.format(sky_rate))
            print('road_rate is {}'.format(road_rate))
            print('tree_rate is {}'.format(tree_rate))

        if not (road_rate > threshold_cp or sky_rate > threshold_cp or tree_rate > threshold_cp):
            continue
        # print("img name: {}".format(img))
        new_name = out_img_dir + img_path[img_path.rfind('/')+1 :]
        render_new_name = render_img_dir + img_path[img_path.rfind('/')+1 :]
        print(img_path);

        shutil.copyfile(img_path, new_name)
        if is_render:
            # print("type: {}".format(type(outs)))
            show_result(img_path, outs, out_file= render_new_name, palette=PALETTE)

        count_num += 1
        print('deal img_path :'.format(count_num), end=' ')

    except :
        raise
    # except Exception as e:
        # print("exception: {}".format(e))
        # continue

print("finished deal {}".format(count_num))

