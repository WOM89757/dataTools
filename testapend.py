import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    'dir_path', default='datesets/img', type=Path, help='Directory path of videos')
args = parser.parse_args()

print(args.dir_path)
class_dir_path = args.dir_path

with open("out.txt","w") as f:

    for video_file_path in sorted(class_dir_path.iterdir()):
        # print(video_file_path.name)
        # ../datasets/img1.1/evalImageSet/image_276.jpg	2
        name = '../datasets/img1.1/evalImageSet/' + video_file_path.name
        print(name)
        f.write(name + ' 2\n')
    # # print(dst_class_path)
    # img_list = os.listdir(video_file_path)
    # for iter in img_list:
    #     # print(iter)
    #     oldfile =  str(video_file_path) + "/" + iter
    #     newfile = '{}/image_{}.jpg'.format(dst_class_path, count)
    #     # print(oldfile)
    #     # print(newfile)
    #     count += 1
    #     shutil.copyfile(oldfile, newfile)