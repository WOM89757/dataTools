import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    'dir_path', default='datesets/img', type=Path, help='Directory path of videos')
parser.add_argument(
    'dst_path', default='datesets/img2', type=Path, help='Directory path of jpg videos')
args = parser.parse_args()

class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
print(class_dir_paths)

dst_dir_path = args.dst_path
dst_dir_path.mkdir(exist_ok=True)



for class_dir_path in class_dir_paths:
    if not class_dir_path.is_dir():
        continue
    # print(class_dir_path.name)
    print(class_dir_path)
    dst_class_path = dst_dir_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)
    
    count = 1
    comm_path =  "datasets/img2/" + class_dir_path.name

    for video_file_path in sorted(class_dir_path.iterdir()):
        # print(video_file_path)
        # print(dst_class_path)
        img_list = os.listdir(video_file_path)
        for iter in img_list:
            # print(iter)
            oldfile =  str(video_file_path) + "/" + iter
            newfile = '{}/image_{}.jpg'.format(comm_path, count)
            
            # print(oldfile)
            # print(newfile)
            count += 1
            shutil.copyfile(oldfile, newfile)

    #     video_process(video_file_path, dst_class_path, ext, fps, size)

print('finished !')