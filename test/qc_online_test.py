import queue
import sys
import threading
import subprocess as sp
from time import sleep
import requests

class Live(object):
    def __init__(self):
        self.push_queue = queue.Queue()
        self.send_queue = queue.Queue()
        self.rtsp_urls = ""
        self.camera_paths = ""
        self.threads_num = 2

    def print_push_stream(self):
        print("child start push stream: {} rtsp: {}".format(self.camera_path[self.camera_path.rfind('/')+1:], self.rtmpUrl))

    def push_stream(self):
        while True:
            if self.push_queue.empty():
                break;
            task = self.push_queue.get()
            camera_path = task[0]
            rtsp_url = task[1]
            # ffmpeg command
            command = ['ffmpeg',
                    '-re',
                    '-i', camera_path,
                    '-vcodec', 'h264', 
                    '-f', 'rtsp', 
                    rtsp_url]
            # print("command {}".format(self.command))
            print("start push stream: {} rtsp: {}".format(camera_path[camera_path.rfind('/')+1:], rtsp_url))
            self.send_queue.put(rtsp_url)
            p = sp.run(command, capture_output=True)
            res = p.stdout.decode('utf-8')
            if p.returncode != 0:
                print(res)
                print("push {} stream fail!".format(rtsp_url))
                continue
            print("end push stream {} ".format(rtsp_url))
        print("thread push stream end")

    def send_requests(self):
        while True:
            try:
                if self.push_queue.empty() and self.send_queue.empty():
                    break;
                sleep(4)
                rtsp_url = self.send_queue.get()

                register_url = "http://127.0.0.1:10008/api/v1/pushers?q="+rtsp_url[rtsp_url.rfind("/"):]
                header = {
                    "Content-Type":"application/json"
                }
                response = requests.get(url=register_url, headers=header)
                # print(response.json())
                if response.json()['total'] == 0:
                    self.send_queue.put(rtsp_url)
                    continue

                # print('rtsp_url: ' + rtsp_url)
                print("start send http requests")

                register_url = "http://127.0.0.1:5053/rmcontext/videodetect"
                header = {
                    "Content-Type":"application/json"
                }
                json_str = {
                    'jkdz':rtsp_url
                }
                response = requests.post(url=register_url, json=json_str, headers=header)
                # print(response.json())
                print(response.json()['sfzx'])
                print(response.json()['sbjg'])
                # print(response.json()['msg'])
            except Exception as e:
                print("send http exception: {}" .format(e))
                sys.exit()
        print("thread send requests end")


            # curl -XPOST http://127.0.0.1:5053/rmcontext/videodetect -H 'Content-Type:application/json' -d '{"jkdz":"rtsp://127.0.0.1:554/test1"}'


    # ffmpeg -re  -i /home/wangmao/dataset/videos/v1.1/zhedang/zhedang-20.mp4 -vcodec h264 -f rtsp  rtsp://127.0.0.1:554/test1


    def run(self):
        try:
            for index in range(len(self.camera_paths)):
                task = [self.camera_paths[index], self.rtsp_urls[index]]
                self.push_queue.put(task)
            print('process task num is {}'.format(self.push_queue.qsize()))
            print('process thread num is {}'.format(self.threads_num))
            threads = []
            [threads.append(threading.Thread(target=Live.push_stream, args=(self,))) for i in range(self.threads_num)]
            # [thread.setDaemon(True) for thread in threads]
            [thread.start() for thread in threads]
            print("process thread start finshed!")
            http_threads = []
            [http_threads.append(threading.Thread(target=Live.send_requests, args=(self,))) for i in range(1)]
            [thread.start() for thread in http_threads]


        except Exception:
            print("run push stream fail!")

if __name__ == '__main__':


    live = Live()
    live.camera_paths = ["/home/wangmao/dataset/videos/v1.1/zhedang/zhedang-20.mp4"
                        # ,"/home/wangmao/dataset/videos/v1.1/zhedang/zhedang-24.mp4"
                        # ,"/home/wangmao/dataset/videos/v1.1/zhedang/zhedang-26.mp4"
                        ]
    live.rtsp_urls= ["rtsp://127.0.0.1:554/" + live.camera_paths[0][live.camera_paths[0].rfind('/')+1:]
                    # ,"rtsp://127.0.0.1:554/" + live.camera_paths[1][live.camera_paths[1].rfind('/')+1:]
                    # ,"rtsp://127.0.0.1:554/" + live.camera_paths[2][live.camera_paths[2].rfind('/')+1:]
                    ]
    live.threads_num = 2
    live.run()
