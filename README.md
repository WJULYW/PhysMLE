# PhysMLE
Official code of TPAMI2025 "PhysMLE: Generalizable and Priors-Inclusive Multi-task Remote Physiological Measurement"

<div style="text-align:center;">
  <img src="framework.png" style="width:100%;" />
</div>


## Data Prepare
You can refer to [link](https://github.com/WJULYW/HSRD) for STMap preprocessing.
Before that, please get the permission to use the following datasets first:
[**VIPL**](http://vipl.ict.ac.cn/en/resources/databases/201901/t20190104_34800.html),
[**V4V**](https://competitions.codalab.org/competitions/31978),
[**BUAA**](https://ieeexplore.ieee.org/document/9320298),
[**UBFC**](https://sites.google.com/view/ybenezeth/ubfcrppg), 
[**PURE**](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure),

[**HCW**] Please scan and dispatch the completed agreement via your institutional email to jwanggo@connect.ust.hk. The email should have the subject line: “HCW Access Request - your institution.” In the email, outline your institution’s past research and articulate the rationale for seeking access to the HCW, including its intended application in your specific research project.

After getting STMaps, you can create a new './STMap' folder and put them into it.

## Pre-trained Model
In this work, we utilized [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and [ViT](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) as the backbone network.
You can download them, create a new folder './pre_encoder' and put the pth file into it.
For the first time running, please adjust the hyperparameter 'reData' to 1, to generate the STMap index.


## Train and Test
Then, you can try to train it with the following command:

```
python train.py -g $GPU id$ -t 'the target dataset you want to test on' -alpha 'lora alpha' -r 'lora gamma' -k 'number of experts'
```
## Please cite following works
```
@ARTICLE{10903997,
  author={Wang, Jiyao and Lu, Hao and Wang, Ange and Yang, Xiao and Chen, Yingcong and He, Dengbo and Wu, Kaishun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={PhysMLE: Generalizable and Priors-Inclusive Multi-Task Remote Physiological Measurement}, 
  year={2025},
  volume={47},
  number={6},
  pages={4908-4925},
  keywords={Multitasking;Biomedical monitoring;Training;Semantics;Blood;Skin;Faces;Videos;Physiology;Heart rate variability;rPPG;multi-task learning;mixture of experts;low-rank adaptation;domain generalization},
  doi={10.1109/TPAMI.2025.3545598}}


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

@ARTICLE{10752618,
  author={Wang, Jiyao and Lu, Hao and Han, Hu and Chen, Yingcong and He, Dengbo and Wu, Kaishun},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Generalizable Remote Physiological Measurement via Semantic-Sheltered Alignment and Plausible Style Randomization}, 
  year={2025},
  volume={74},
  number={},
  pages={1-14},
  keywords={Heart rate;Uncertainty;Volume measurement;Semantics;Estimation;Photoplethysmography;Feature extraction;Skin;Robustness;Biomedical monitoring;Domain generalization (DG);heart rate (HR) estimation;invariant risk minimization (IRM);plausible style generation;remote photoplethysmography (rPPG)},
  doi={10.1109/TIM.2024.3497058}}



```
