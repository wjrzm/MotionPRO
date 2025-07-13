# MotionPRO

## Introduction

<h1 align='Center'>MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond</h1>
<div align='Center'>
    <a href='https://www.wjrzm.com' target='_blank'>Shenghao Ren</a><sup>1*</sup>&emsp;
    <a href='https://yeelou.github.io/' target='_blank'>Yi Lu</a><sup>1*</sup>&emsp;
    Jiayi Huang<sup>1</sup>&emsp;
    Jiayi Zhao<sup>1</sup>&emsp;
    <br>
    He Zhang<sup>3</sup>&emsp;
    <a href='https://ytrock.com' target='_blank'>Tao Yu</a><sup>3</sup>&emsp;
    <a href='https://shenqiu.njucite.cn/' target='_blank'>Qiu Shen</a><sup>1,2†</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1,2</sup>
</div>
<div align='Center'>
    <sup>1</sup>School of Electronic Science and Engineering, Nanjing University, Nanjing, China 
    <br>
    <sup>2</sup>Key Laboratory of Optoelectronic Devices and Systems with Extreme
        <br>
    Performances of MOE, Nanjing University, Nanjing, China 
    <br>
    <sup>3</sup>BNRist, Tsinghua University, Beijing, China
</div>
<div align='Center'>
    <sup>*</sup>Equal Contribution
    <sup>†</sup>Corresponding Author
</div>
<div align='Center'>
    <a href='https://nju-cite-mocaphumanoid.github.io/MotionPRO/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2504.05046'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://www.youtube.com/watch?v=UkUj3kiR5ss'><img src='https://badges.aleen42.com/src/youtube.svg'></a>

</div>


![pipeline](https://image.wjrzm.com/i/2025/05/25/ytk0dx.png)

## Installation
Create conda environment:

```bash
  conda create -n motionpro python=3.8
  conda activate motionpro
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
```

1. Download [SMPL](https://smpl.is.tue.mpg.de/) models. Rename neutral(version 1.1.0) to `SMPL_NEUTRAL.pkl`. And move `SMPL_NEUTRAL.pkl` to `data/smpl`.

2. Download [CLIFF](https://drive.google.com/drive/folders/1dAZiPqJY2wBv6QzpjOwYi4Ax1y-oBIM1) checkpoint `hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt`. And move `hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt` to `data/cliff_ckpt`.

3. Download [YOLOX](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth). And move `yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth` to `data/mmdetection/checkpoints`.

## Data preparation

MotionPRO Dataset is available at https://shenqiu.njucite.cn/download

1. Generate bounding box for each image: 
```bash
  python -m lib.util.gen_bbox
```

2. Generate image feature for each image by using bbox:
```bash
  python -m lib.util.gen_image_feature
```

3. Generate keypoints:
```bash
  python -m lib.util.gen_kps
```

4. Generate contact:
```bash
  python -m lib.util.gen_contact
```

## Train
Using all the prepared data for training FRAPPE:
```bash
python -m app.train_frappe
```

## Citation

If you find this code useful, please consider citing:

```
@inproceedings{ren2025motionpro,
  title={MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond},
  author={Ren, Shenghao and Lu, Yi and Huang, Jiayi and Zhao, Jiayi and Zhang, He and Yu, Tao and Shen, Qiu and Cao, Xun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={27760--27770},
  year={2025}
}
```

## Contact

If you have any questions, please contact: Shenghao Ren (shenghaoren@smail.nju.edu.cn) or Yi Lu (yi.lu@smail.nju.edu.cn).
