# formConvert
> find "####" line data, split by last '/'

```bash
    python testDataSplit.py --source inputtest.txt --target resulttest.txt
```


# video2jpeg

> 转换视频数据为图片数据集，若存在test目录，安装类别分开存放数据集

- video convert

```bash
    python generate_video_jpgs.py /home/wangmao/dataset/videos/ /home/wangmao/dataset/img zhedang
```

- merge date

> 将数据按照训练和测试集进行合并 

```bash
    python merge_data.py /home/wangmao/dataset/img1/ ../../datasets/img1.1/
```