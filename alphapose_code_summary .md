# Alphapose Summary
## 1.Paper
### 1.1 download <img src="https://github.com/wmj142326/Img_packag/raw/alphapose/2.baidudownload.png" width="3%"/>

* paper link:  [RMPE：**R**egional **M**ulti-Person **P**ose **E**stimation](https://pan.baidu.com/s/1L1sAJQVqigGSewF_QrZ6_w ). 

* passward:  **dwxd** 
### 1.2 abstract

* <font color="#4590a3" size="4px">**SSTN:**  </font>Symmetric  Spatial Transformer Network                         （对称空间网络变换）：`精确提取单人姿态` 

* <font color="#4590a3" size="4px">**p-Pose NMS:**  </font>Parametric Pose NonMaximum-Suppression    （参数化姿态非极大抑制）：`解决冗余 `

* <font color="#4590a3" size="4px">**PGPG:** </font>Pose-Guided Proposals Generator                                    （姿态引导区域框生成器）：`增强训练数据`  

  <img src="https://github.com/wmj142326/Img_packag/raw/alphapose/3.RMPE.png " width="60%">
---

## 2.demo

### 2.1 download<img src="https://github.com/wmj142326/Img_packag/raw/alphapose/4.github.png" width="3%">

* code link1: [AlphaPose-Old](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) 
* code link2:[Alphapose-New](https://github.com/MVIG-SJTU/AlphaPose)

### 2.2 Train

#### 2.2.1 dataset path modification

```python
"""
自己训练时首先要修改MScoco数据集路径：
"""

# ./alphapose/utils/metrics.py 
ann_file = os.path.join('./data/coco/annotations/', ann_file)

# /alphapose/datasets/coco_det.py  
img_path = './data/coco/val2017/%012d.jpg' % img_id

# /configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
DATASET:
  TRAIN:
    ROOT: './data/coco/'
  VAL:
    ROOT: './data/coco/'
  TEST:
    ROOT: './data/coco/'
        
        
"""
训练指令：
"""
> python ./scripts/train.py --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --exp-id exp_fastpose
```

#### 2.2.2 download<img src="https://github.com/wmj142326/Img_packag/raw/alphapose/2.baidudownload.png" width="3%"/>

训练好的模型`.pth`文件将在生成的**exp**文件夹内，包括训练日志`train.log`和可视化训练数据`tensorboard`：

* train link : [exp(部分)](https://pan.baidu.com/s/1UiXdeLDMDomQqB7vA4obBA  )

* passward: **3d9k** 

### 2.3 Test

```python
# picture：图片测试
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint 模型路径/final_DPG.pth --indir 图片文件夹路径 --outdir out/pic/pic_4 --save_img --detbatch 1 --posebatch 30
    
# video：视频测试
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint 模型路径/final_DPG.pth --video 视频路径/文件名.mp4 --outdir 输出文件夹路径 --save_video --detbatch 1 --posebatch 30
    
# camera 相机测试
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint 模型路径/final_DPG.pth --outdir out/webcam_1 --vis --webcam 0
    
# eg 示例
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint input/exp/DPG_AlphaPose/final_DPG.pth --indir examples/demo/pic/pic_4 --outdir out/pic/pic_4_DPG --save_img --detbatch 1 --posebatch 30
```

## 3.code

### 3.1 download<img src="https://github.com/wmj142326/Img_packag/raw/alphapose/2.baidudownload.png" width="3%"/>

Alphapose代码的部分注释在下面链接，可随时**更新**下载：

* alphapose link:[Alphapose](https://pan.baidu.com/s/1Q7QpvhjanWYZ0b0GxQU6hA)

* passward: **18re **

  注释部分主要包括以下`alphapose`文件夹和`train.py`中**main()**函数的部分:

<img src="https://github.com/wmj142326/Img_packag/raw/alphapose/5.alphapose_xmind.png" width="100%"/>

### 3.2 three parts

#### 3.2.1 SSTN

```python
# transformer.py

# 463
def drawGaussian(img, pt, sigma):
    """
    Draw 2d gaussian on input image.
    """
    ......
    
# 587
def heatmap_to_coord_simple(hms, bbox, hms_flip=None, **kwargs):
    ......
```



#### 3.2.2 p-Pose NMS

<img src="https://github.com/wmj142326/Img_packag/raw/alphapose/7.pPose-NMS.png" width="60%"/>

```python
# pPose_nms

#200
def pose_nms(bboxes, bbox_scores, bbox_ids, pose_preds, pose_scores, areaThres=0):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n, 1)  各个框为人的score
    bbox_ids:       bbox tracking ids list (n, 1)
    pose_preds:     pose locations list (n, kp_num, 2)  各关节点的坐标
    pose_scores:    pose scores list    (n, kp_num, 1)  各个关节点的score
    '''
	......
    
# 432
def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    
    ......
    
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.repeat(1, all_preds.shape[0]).transpose(0, 1)

    score_dists[mask] = torch.tanh(pred_scores[mask] / delta1) * torch.tanh(keypoint_scores[mask] / delta1)  # delta1 = 1，当前点和近距离点的score的相似度，公式（8）

    point_dist = torch.exp((-1) * dist / delta2)  # # delta2 = 2.65，当前点和近距离点的距离的相似度，公式（9）
    final_dist = torch.sum(score_dists, dim=1) + mu * torch.sum(point_dist, dim=1) # mu = 1.7，最终的距离 

    return final_dist
```

#### 3.3.3 PGPG

```python
# transformer.py

# 43
def addDPG(bbox, imgwidth, imght):
    """Add dpg for data augmentation, including random crop and random sample."""
```

## 4.Note

* [(原)人体姿态识别alphapose](https://www.cnblogs.com/darkknightzh/p/12150171.html#_lab2_3_2)
* [pytorch模型的保存与加载](https://www.jianshu.com/p/60fc57e19615)
* [Python @函数装饰器及用法](http://c.biancheng.net/view/2270.html)
* [Python模块之Logging ——常用handlers的使用](https://blog.csdn.net/wangpengfei163/article/details/80423863)

* [调整学习率-torch.optim.lr_scheduler.MultiStepLR（）方法](https://www.cnblogs.com/shuangcao/p/12127506.html)