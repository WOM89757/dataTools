import random
import argparse
from pathlib import Path
import shutil
import codecs
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    'dir_path', default='datesets/img', type=Path, help='Directory path of videos')
parser.add_argument(
    'dst_path', default='datesets/splitimg', type=Path, help='Directory path of jpg videos')
parser.add_argument(
    '--r', default=0.7, type=float, help='train ratio')
args = parser.parse_args()

dst_dir_path = args.dst_path
dst_dir_path.mkdir(exist_ok=True)
split_img_flag = True
split_info_file = codecs.open(os.path.join(dst_dir_path, "split_info.txt"), 'w')
random.seed(2)

def split(all_list, shuffle=False, ratio=0.8):
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0:
        return all_list, []
    if shuffle:
        random.shuffle(all_list)
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test

train_ratio = args.r
split_info_file.write("data split ratio is {0}\n".format(train_ratio))

for class_dir_paths in sorted(args.dir_path.iterdir()):
    print('current path: {}'.format(class_dir_paths))
    class_name = str(class_dir_paths)[str(class_dir_paths).rfind('/')+1:]
    train_dst_class_path = str(dst_dir_path) + '/train/' + class_name
    test_dst_class_path = str(dst_dir_path) + '/test/' + class_name
    Path(train_dst_class_path).mkdir(exist_ok=True, parents = True)
    Path(test_dst_class_path).mkdir(exist_ok=True, parents = True)

    # print('class_name:{}'.format(class_name))
    # print(class_dir_paths.iterdir().size())
    video_imgs = [p for p in Path(class_dir_paths).iterdir() if p.is_dir]
    video_imgs.sort()
    # print('name:{}'.format(len(video_imgs)))
    traindatas, testdatas = split(video_imgs, shuffle=True, ratio=train_ratio)
    # print('train({}): {}'.format(len(traindatas), traindatas))
    # print('testdatas({}): {}'.format(len(testdatas), testdatas))
    train_count = 0
    test_count = 0
    if split_img_flag:
        for img_path in traindatas:
            file_name = str(img_path)[str(img_path).rfind('/')+1:]
            dst_path = train_dst_class_path + '/' + file_name
            # print("video_dir_path: {}".format(img_path))
            # print("img_dst_path: {}".format(dst_path))
            shutil.copy(img_path, dst_path)
            train_count += 1
        
        for img_path in testdatas:
            file_name = str(img_path)[str(img_path).rfind('/')+1:]
            dst_path = test_dst_class_path + '/' + file_name
            # print("video_dir_path: {}".format(img_path))
            # print("img_dst_path: {}".format(dst_path))
            shutil.copy(img_path, dst_path)
            test_count += 1
    else:
        for video_path in traindatas:
            for img_path in Path(video_path).iterdir():
                file_name = str(img_path)[str(img_path).rfind('/')+1:]
                dst_path = train_dst_class_path + '/' + file_name
                # print("video_dir_path: {}".format(img_path))
                # print("img_dst_path: {}".format(dst_path))
                shutil.copy(img_path, dst_path)
                train_count += 1
        
        for video_path in testdatas:
            for img_path in Path(video_path).iterdir():
                file_name = str(img_path)[str(img_path).rfind('/')+1:]
                dst_path = test_dst_class_path + '/' + file_name
                # print("video_dir_path: {}".format(img_path))
                # print("img_dst_path: {}".format(dst_path))
                shutil.copy(img_path, dst_path)
                test_count += 1
    
    
    split_info = '{}({}) train({}): {}, test({}): {}'.format(class_name, len(video_imgs), len(traindatas), train_count, len(testdatas), test_count)
    print(split_info)
    split_info_file.write("{0}\n".format(split_info))

    # print('{}({}) train({}): {}, test({}): {}'.format(class_name, len(video_imgs), len(traindatas), train_count, len(testdatas), test_count))
    
split_info_file.close()
print("finshed split!")
