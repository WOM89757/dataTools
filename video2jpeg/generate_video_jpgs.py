import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    # print(video_file_path)

    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    # print(res)

    if len(res) < 4:
        return

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])
    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    # dst_dir_path = dst_root_path
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    # print("class_dir_path is " + str(class_dir_path))
    if not class_dir_path.is_dir():
        return
    if 'test/' in str(class_dir_path) :
        dst_class_path = dst_root_path / 'test' / class_dir_path.name
    else:
        dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        # print(str(video_file_path) + ' ---- ' + str(dst_class_path))

        video_process(video_file_path, dst_class_path, ext, fps, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        '-d',
        '--dataset',
        default='zhedang',
        type=str,
        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet)')
    parser.add_argument(
        '--n_jobs', default=4, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=640, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['kinetics', 'mit', 'activitynet', 'zhedang', 'qualitycheck']:
        ext = '.mp4'
    else:
        ext = '.avi'
    print(args)
    if args.dataset in ['activitynet']:
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps, args.size)
                                 for video_file_path in video_file_paths)
    else:
        class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
        # print(class_dir_paths)
        test_set_video_path = args.dir_path / 'test'
        test_set_video_paths = Path(test_set_video_path)
        # print(test_set_video_paths)
        if test_set_video_path.exists():
            test_class_dir_paths = [x for x in sorted(test_set_video_paths.iterdir())]
            # print(test_class_dir_paths)
            for iter in test_class_dir_paths:
                class_dir_paths.append(test_set_video_path / iter.name)
        
        args.dst_path.mkdir(exist_ok=True)
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in class_dir_paths)
    
    print("finished convert !")
