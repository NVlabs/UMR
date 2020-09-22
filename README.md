# Self-supervised Single-view 3D Reconstruction via Semantic Consistency

### [project](https://sites.google.com/nvidia.com/unsup-mesh-2020) | [paper](https://arxiv.org/abs/2003.06473)

[Xueting Li](https://sunshineatnoon.github.io/) &nbsp; [Sifei Liu](https://www.sifeiliu.net/) &nbsp; [Kihwan Kim](http://www.kihwan23.com/) &nbsp; [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta) &nbsp; [Varun Jampani](https://varunjampani.github.io/) &nbsp; [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/) &nbsp; [Jan Kautz](http://jankautz.com/)

NVIDIA &nbsp; UC Merced

ECCV 2020

![](docs/images/teaser.gif)

## Citation
Please cite our paper if you find this code useful for your research.
```
@inproceedings{umr2020,
  title={Self-supervised Single-view 3D Reconstruction via Semantic Consistency},
  author={Li, Xueting and Liu, Sifei and Kim, Kihwan and De Mello, Shalini and Jampani, Varun and Yang, Ming-Hsuan and Kautz, Jan},
  booktitle={ECCV},
  year={2020}
}
```

## Prerequisites

- Download code & pre-trained model:

  Git clone the code by:
  ```
  git clone https://github.com/NVlabs/UMR $ROOTPATH
  ```
  We are working on to release the pre-trained model, please contact the authors for more details.

- Install packages:

  [Virtual environments](https://docs.python.org/3/tutorial/venv.html) are not required but highly recommended:
  ```
  conda create -n umr python=3.6
  source activate umr
  ```
  Then run:
  ```
  cd $ROOTPATH/UMR
  sh install.sh
  ```

## Run the demo
Run the following command from the `$ROOTPATH` directory:
```
python -m UMR.experiments.demo --model_path UMR/cachedir/snapshots/cub_net/pred_net_latest.pth --img_path UMR/demo_imgs/birdie.jpg --out_path UMR/cachedir/demo
```
The results will be saved at `out_path` in the above command.

## Quantitative Evaluation
Download CUB bird images by:
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```

Command for keypoint transfer evaluation using predicted texture flow:
```
python -m UMR.experiments.test_kp --model_path UMR/cachedir/snapshots/cub_net/pred_net_latest.pth --split test --number_pairs 10000 --cub_cache_dir UMR/cachedir/cub/ --cub_dir CUB_200_2011/
```

Command for keypoint transfer evaluation using predicted camera pose:
```
python -m UMR.experiments.test_kp --model_path UMR/cachedir/snapshots/cub_net/pred_net_latest.pth --split test --number_pairs 10000 --mode cam --cub_cache_dir UMR/cachedir/cub/ --cub_dir CUB_200_2011/
```
For keypoint transfer result visualization, add `--visualize`. The visualizations will be saved at `$ROOTPATH/UMR/cachedir/results_vis/`

Command for evaluating mask IoU:
```
python -m UMR.experiments.test_iou --model_path UMR/cachedir/snapshots/cub_net/pred_net_latest.pth --split test --cub_cache_dir UMR/cachedir/cub/ --cub_dir CUB_200_2011/ --batch_size 32
```

## Training
To train the full model with semantic consistency constraint:
- Prepare SCOPS predictions following instructions [here](https://github.com/NVlabs/SCOPS#scops-on-caltech-ucsd-birds).
- Download semantic template as described [above](https://github.com/NVlabs/UMR#prerequisites) if haven't done so.
- Run the following command from the `$ROOTPATH` folder:

  ```
  python -m UMR.experiments.train_s2 --name=cub_s2 --batch_size 16 --cub_dir CUB_200_2011/ --cub_cache_dir UMR/cachedir/cub --scops_path SCOPS/results/cub/ITER_60000/train/dcrf_prob/ --stemp_path UMR/cachedir/cub/scops/
  ```
  If the code is ran on multiple GPUs, add `--multi_gpu --gpu_num GPU_NUMBER` to the above commands.

If you wish to train from scratch (i.e. learn the semantic template from scratch):
- Run the following command from the `$ROOTPATH` folder:

  ```
  python -m UMR.experiments.train_s1 --name=cub_s1 --gpu_num 4 --multi_gpu -cub_dir CUB_200_2011/ --cub_cache_dir UMR/cachedir/cub
  ```
- Then compute the semantic template by:
  ```
  python -m UMR.experiments.avg_uv --model_path UMR/cachedir/snapshots/cub_s1/pred_net_latest.pth --batch_size 16 --out_dir UMR/cachedir/snapshots/cub_s1 --cub_dir CUB_200_2011/ --cub_cache_dir UMR/cachedir/cub --scops_path SCOPS/results/cub/ITER_60000/train/dcrf_prob/
  ```
  Now you can use the computed semantic template in the model training by setting `stemp_path` in `train_s2.py` to `out_dir` in the above command.

## License
Unless otherwise indicated, all code is Copyright (C) 2020 NVIDIA Corporation. All rights reserved. Licensed under the NVIDIA Source Code License-NC.

## Acknowledgement
This code is built on [CMR](https://github.com/akanazawa/cmr), [CSM](https://github.com/nileshkulkarni/csm), [project_skeleton](https://github.com/shubhtuls/project_skeleton). We modified the [SoftRas](https://github.com/ShichenLiu/SoftRas) for the proposed texture cycle consistency objective and the [PerceptualSimilarity](https://github.com/shubhtuls/PerceptualSimilarity) to support multiple GPU training. We included the modified codes in the `external` folder for convenience. If you find these modules useful, please cite the corresponding paper.

