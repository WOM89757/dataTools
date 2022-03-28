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

# class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
# print(class_dir_paths)

dst_dir_path = args.dst_path
dst_dir_path.mkdir(exist_ok=True)


for class_dir_paths in sorted(args.dir_path.iterdir()):
    # print(dir_class)
    # for class_dir_paths in sorted(dir_class.iterdir()):
    # print(class_dir_paths)
    new_dst_path = dst_dir_path / class_dir_paths.name
    # print(new_dst_path)
    new_dst_path.mkdir(exist_ok=True)


    for class_dir_path in sorted(class_dir_paths.iterdir()):
        if not class_dir_path.is_dir():
            continue
        # print(class_dir_path.name)

        end_pos = str(class_dir_path).rfind('/') - 1  
        start_pos = str(class_dir_path).rfind('/', 0, end_pos)  
        prefix_name = str(class_dir_path)[start_pos + 1:]

        dst_class_path = dst_dir_path / prefix_name
        dst_class_path.mkdir(exist_ok=True)
        
        count = 1

        for video_file_path in sorted(class_dir_path.iterdir()):
            # print(video_file_path)
            # print(dst_class_path)
            img_list = os.listdir(video_file_path)
            for iter in img_list:
                # print(iter)
                oldfile =  str(video_file_path) + "/" + iter
                newfile = '{}/image_{}.jpg'.format(dst_class_path, count)
                # print(oldfile)
                # print(newfile)
                count += 1
                shutil.copyfile(oldfile, newfile)

print('finished !')