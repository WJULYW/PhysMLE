# PhysMLE
Official code of "PhysMLE: Generalizable and Priors-Inclusive Multi-task Remote Physiological Measurement"

## Data Prepare
You can refer to [https://github.com/EnVision-Research/NEST-rPPG](https://github.com/WJULYW/HSRD).
Before that, please get the permission to use the following datasets first:
[**VIPL**](http://vipl.ict.ac.cn/en/resources/databases/201901/t20190104_34800.html),
[**V4V**](https://competitions.codalab.org/competitions/31978),
[**BUAA**](https://ieeexplore.ieee.org/document/9320298),
[**UBFC**](https://sites.google.com/view/ybenezeth/ubfcrppg), 
[**PURE**](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure).
[**HCW**] If you want to use this dataset, please send an email to jwang297@connect.hkust-gz.edu.cn with the Title of 'HCW Dataset Usage Application' and a brief statement about your usage purpose.
After getting STMaps, you can create a new './STMap' folder and put them into it.

## Pre-trained Model
In this work, we utilized ResNet18[link](https://download.pytorch.org/models/resnet18-5c106cde.pth) and ViT[link](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) as the backbone network.
You can download them, create a new folder './pre_encoder' and put the pth file into it.
For the first time running, please adjust the hyperparameter 'reData' to 1, to generate the STMap index.


## Train and Test
Then, you can try to train it with the following command:

```
python train.py -g $GPU id$ -t 'the target dataset you want to test on' -alpha 'lora alpha' -r 'lora gamma' -k 'number of experts'
```
## Please cite following works
```
@article{wang2024physmle,
  title={PhysMLE: Generalizable and Priors-Inclusive Multi-task Remote Physiological Measurement},
  author={Wang, Jiyao and Lu, Hao and Wang, Ange and Yang, Xiao and Chen, Yingcong and He, Dengbo and Wu, Kaishun},
  journal={arXiv preprint arXiv:2405.06201},
  year={2024}
}

@ARTICLE{10371379,
  author={Wang, Jiyao and Lu, Hao and Wang, Ange and Chen, Yingcong and He, Dengbo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Hierarchical Style-Aware Domain Generalization for Remote Physiological Measurement}, 
  year={2024},
  volume={28},
  number={3},
  pages={1635-1643},
  keywords={Feature extraction;Videos;Skin;Physiology;Biomedical measurement;Bioinformatics;Training;Adversarial learning;contrastive learning;domain generalization;heart rate estimation;remote photoplethysmography (rPPG)},
  doi={10.1109/JBHI.2023.3346057}}

@ARTICLE{dg2024wang,
  author={Wang, Jiyao and Lu, Hao and Han, Hu and Chen, Yingcong and He, Dengbo and Wu, Kaishun},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Generalizable Remote Physiological Measurement via Semantic-Sheltered Alignment and Plausible Style Randomization}, 
  year={2024}
}


```
