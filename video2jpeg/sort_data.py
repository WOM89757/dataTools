import argparse
import os
import shutil
from pathlib import Path
from xxlimited import new

parser = argparse.ArgumentParser()
parser.add_argument(
    'dir_path', default='datesets/img', type=Path, help='Directory path of videos')
parser.add_argument(
    'dst_path', default='datesets/img2', type=Path, help='Directory path of jpg videos')
args = parser.parse_args()

dst_dir_path = args.dst_path
dst_dir_path.mkdir(exist_ok=True)
    
count = 1

for video_file_path in sorted(args.dir_path.iterdir()):
    if video_file_path.is_dir():
        continue
    # print(video_file_path)
    # print(dst_class_path)
    oldfile =  str(video_file_path)
    end_pos = str(video_file_path).rfind('.')
    suffix_name = str(video_file_path)[end_pos + 1:]
    # print(suffix_name)
    newfile = '{}/{}-{}.{}'.format(dst_dir_path, args.dir_path.name, count, suffix_name)
    # print(oldfile)
    # print(newfile)
    count += 1
    shutil.copyfile(oldfile, newfile)

print('finished !')