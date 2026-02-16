# Awesome-Scene-Change-Detection(SCD)

This repo contains a curative list of **scene change detection(SCD)**, including papers, videos, codes, and related websites. 

_NOTE. This repository covers scene change detection based on robot vision (Especially image and point cloud data, etc.). Many change detection papers focus on the remote sensing domain, but this repo will only list works that have been tested on street-view scenes._

#### Please feel free to send me [pull requests](https://github.com/DoongLi/Awesome-Scene-Change-Detection/blob/main/how-to-PR.md) or [email](mailto:lidong8421bcd@gmail.com) to add papers! 

If you find this repository useful, please consider STARing this list. Feel free to share this list with others!

There are some similar repositories available, but it appears that they have not been maintained for a long time.

- https://github.com/wenhwu/awesome-remote-sensing-change-detection
- https://github.com/chickenbestlover/awesome-scene-change-detections
- https://github.com/MinZHANG-WHU/Change-Detection-Review

## Paper

#### 2026

- Chamelion: Reliable Change Detection for Long-Term LiDAR Mapping in Transient Environments, `RAL`. [[Paper](https://arxiv.org/pdf/2602.08189)] [[Code](https://github.com/OREOCHIZ/chamelion)] [[Website](https://chamelion-pages.github.io/)]
- LiDAR-based 3D Change Detection at City Scale, `arXiv`. [[Paper](https://arxiv.org/pdf/2510.21112)] [[Code](https://github.com/HaitianWang/IEEE-Sensor-Journal-Changing-Detection)]

#### 2025

- SceneDiff: A Benchmark and Method for Multiview Object Change Detection, `arXiv`, [[Paper](https://arxiv.org/pdf/2512.16908)][[Code](https://github.com/yuqunw/scene_diff)]
- Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms, `ICRA`. [[Paper](https://arxiv.org/pdf/2409.16850)] [[Code](https://github.com/ChadLin9596/Robust-Scene-Change-Detection)] [[Website](https://chadlin9596.github.io/projects/2024-image-cd.html)]
- **3DGS-CD**: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement, `RAL`. [[Paper](https://arxiv.org/pdf/2411.03706)] [[Code](https://github.com/520xyxyzq/3DGS-CD)]
- **Gaussian Difference**: Find Any Change Instance in 3D Scenes, `arXiv`. [[Paper](https://arxiv.org/pdf/2502.16941)]
- Towards Generalizable Scene Change Detection, `CVPR`. [[Paper](https://arxiv.org/pdf/2409.06214)] [[Code](https://github.com/1124jaewookim/towards-generalizable-scene-change-detection)]
- ![paper](https://img.shields.io/badge/Dataset-red) **MV-3DCD**: Multi-View Pose-Agnostic Change Localization with Zero Label, `CVPR`. [[Paper](https://arxiv.org/pdf/2412.03911)] [[Website](https://chumsy0725.github.io/MV-3DCD/)]
- EMPLACE: Self-Supervised Urban Scene Change Detection, `AAAI`. [[Paper](https://arxiv.org/pdf/2503.17716)] [[Code](https://github.com/Timalph/EMPLACE)]
- Zero-Shot Scene Change Detection, `AAAI`. [[Paper](https://arxiv.org/pdf/2406.11210)] [[Code](https://github.com/kyusik-cho/ZSSCD)]
- Environmental Change Detection: Toward a Practical Task of Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/2506.11481)]
- Information-Bottleneck Driven Binary Neural Network for Change Detection, `ICCV`. [[Paper](https://arxiv.org/pdf/2507.03504)]
- LT-Gaussian: Long-Term Map Update Using 3D Gaussian Splatting for Autonomous Driving, `arXiv`. [[Paper](https://arxiv.org/pdf/2508.01704)]
- GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments, `ICCV`. [[Paper](https://arxiv.org/pdf/2508.08867)] [[Website](https://zju3dv.github.io/GaussianUpdate)]
- Leveraging Geometric Priors for Unaligned Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/2509.11292)] [[Code](https://github.com/ZilingLiu/GeoSCD)]
- ChangingGrounding: 3D Visual Grounding in Changing Scenes, `arXiv`. [[Paper](https://arxiv.org/pdf/2510.14965)] [[Website](https://hm123450.github.io/CGB/)] [[Code](https://github.com/hm123450/ChangingGroundingBenchmark)]
- Changes in Real Time: Online Scene Change Detection with Multi-View Fusion, `arXiv`. [[Paper](https://arxiv.org/pdf/2511.12370)] [[Code](https://github.com/Chumsy0725/O-SCD)] [[Website](https://chumsy0725.github.io/O-SCD)]
- Spectral-Temporal Attention for Robust Change Detection, `IROS`. [[Paper](https://ieeexplore.ieee.org/document/11246398)]

#### 2024
- ![paper](https://img.shields.io/badge/Dataset-red) **UMAD**: University of Macau Anomaly Detection Benchmark Dataset, `IROS`. [[Paper](https://arxiv.org/pdf/2408.12527)] [[Dataset](https://github.com/IMRL/UMAD)] [[Website](https://doongli.github.io/umad/)]
- **CDMamba**: Remote Sensing Image Change Detection with Mamba, `arXiv`. [[Paper](https://arxiv.org/pdf/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)]
- Semi-Supervised Scene Change Detection by Distillation from Feature-metric Alignment, `WACV`. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Semi-Supervised_Scene_Change_Detection_by_Distillation_From_Feature-Metric_Alignment_WACV_2024_paper.pdf)]
- Change of Scenery: Unsupervised LiDAR Change Detection for Mobile Robots, `arXiv`. [[Paper](https://arxiv.org/pdf/2309.10924)]
  - Keyword: Point clonds change detection;
- **LaserSAM**: Zero-Shot Change Detection Using Visual Segmentation of Spinning LiDAR, `arXiv`. [[Paper](https://arxiv.org/pdf/2402.10321)]
  - Keyword: Point clonds change detection;
- ![paper](https://img.shields.io/badge/Dataset-red) **CityPulse**: Fine-Grained Assessment of Urban Change with Street View Time Series, `arXiv`. [[Paper](https://arxiv.org/pdf/2401.01107v2)]
- Towards Generalizable Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/2409.06214)]
- ZeroSCD: Zero-Shot Street Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/2409.15255)]
- **LiSTA**: Geometric Object-Based Change Detection in Cluttered Environments, `ICRA`. [[Paper](https://ieeexplore.ieee.org/document/10610102)]
- **ViewDelta**: Text-Prompted Change Detection in Unaligned Images, `arXiv`. [[Paper](https://arxiv.org/pdf/2412.07612)]
- Indoor Scene Change Understanding (SCU): Segment, Describe, and Revert Any Change, `IROS`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10801354)]
- ![paper](https://img.shields.io/badge/Dataset-red) **The STVchrono Dataset**: Towards Continuous Change Recognition in Time, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_The_STVchrono_Dataset_Towards_Continuous_Change_Recognition_in_Time_CVPR_2024_paper.pdf)]

#### 2023

- ![paper](https://img.shields.io/badge/Dataset-red) The Change You Want to See (Now in 3D), `ICCVw`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Sachdeva_The_Change_You_Want_to_See_Now_in_3D_ICCVW_2023_paper.pdf)] [[Code](https://github.com/ragavsachdeva/CYWS-3D)]
- ![paper](https://img.shields.io/badge/Dataset-red) The Change You Want to See, `WACV`. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Sachdeva_The_Change_You_Want_To_See_WACV_2023_paper.pdf)] [[Code](https://github.com/ragavsachdeva/The-Change-You-Want-to-See)] [[Website](https://www.robots.ox.ac.uk/~vgg/research/cyws/)]
- How to reduce change detection to semantic segmentation, `PR`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320323000857)] [[Code](https://github.com/DoctorKey/C-3PO)]
  - Keyword: C-3PO;
- **Changes-Aware Transformer**: Learning Generalized Changes Representation, `arXiv`.[[Paper](https://arxiv.org/pdf/2309.13619)]
- **SCTF-Det**: Siamese Center-Based Detector with Transformer and Feature Fusion for Object-Level Change Detection, `CAC`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10451045)]
- **SoftMatch Distance**: A Novel Distance for Weakly-Supervised Trend Change Detection in Bi-Temporal Images, `arXiv`. [[Paper](https://arxiv.org/pdf/2303.04737)]
  - Keyword: General change detection (GCD), trend change detection (TCD);
- Has Anything Changed? 3D Change Detection by 2D Segmentation Masks, `arXiv`. [[Paper](https://arxiv.org/pdf/2312.01148)] [[Code](https://github.com/katadam/ObjectChangeDetection)]
  - Keyword:  3D change detection;
- **Changes-Aware Transformer**: Learning Generalized Changes Representation, `arXiv`. [[Paper](https://arxiv.org/pdf/2309.13619)]
- **3D VSG**: Long-term Semantic Scene Change Prediction through 3D Variable Scene Graphs, `ICRA`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10161212)]
  - Keyword:  3D change detection;
- **ScaleMix**: Intra- And Inter-Layer Multiscale Feature Combination for Change Detection, `ICASSP`. [[Paper](https://ieeexplore.ieee.org/document/10095962)]

#### 2022

- Objects Can Move: 3D Change Detection by Geometric Transformation Constistency, `ECCV`. [[Paper](https://arxiv.org/pdf/2208.09870)] [[Code](https://github.com/katadam/ObjectsCanMove)]
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Dual Task Learning by Leveraging Both Dense Correspondence and Mis-Correspondence for Robust Change Detection With Imperfect Matches, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf)] [[Code](https://github.com/SAMMiCA/SimSaC)]
- `IDET`: Iterative Difference-Enhanced Transformers for High-Quality Change Detection, [[Paper]()] [[Code](https://arxiv.org/pdf/2207.09240)]
- Selecting change image for efficient change detection, `IET Signal Processing`. [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/sil2.12095)]
- Scene change detection: semantic and depth information, `Multimedia Tools and Applications`. [[Paper](https://link.springer.com/article/10.1007/s11042-021-10793-4)]
- Industrial Scene Change Detection using Deep Convolutional Neural Networks, `arXiv`. [[Paper](https://arxiv.org/pdf/2212.14278)]
- Detecting Object-Level Scene Changes in Images with Viewpoint Differences Using Graph Matching, `Remote Sensing`. [[Paper](https://www.mdpi.com/2072-4292/14/17/4225)]
- **Standardsim**: A synthetic dataset for retail environments, International Conference on Image Analysis and Processing. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-06430-2_6)] [[Code](https://github.com/nicholaslocascio/Standard-Sim)]
- A motion-appearance-aware network for object change detection, `Knowledge-Based Systems`. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705122008139)]
- Scene Independency Matters: An Empirical Study of Scene Dependent and Scene Independent Evaluation for CNN-Based Change Detection, `TITS`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9238403)]
- PlaneSDF-Based Change Detection for Long-Term Dense Mapping, `RAL`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9832467)]

#### 2021

- ![paper](https://img.shields.io/badge/Dataset-red) **Changesim**: Towards end-to-end online scene change detection in industrial indoor environments, `IROS`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636350)] [[Code](https://github.com/SAMMiCA/ChangeSim)] [[Website](https://sammica.github.io/ChangeSim/)]
- Hierarchical Paired Channel Fusion Network for Street Scene Change Detection, `TIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9246289)]
- **DR-TANet**: Dynamic Receptive Temporal Attention Network for Street Scene Change Detection, `IV`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9575362&casa_token=dtDExlOCcvsAAAAA:0DGIqppF3PvXwlT5fC9KhyNz-3s7sxNZu8FpyMx6L1Lgeh5QMB_yETx0EN9ax5TaR7djpJ910w)] [[Code](https://github.com/Herrccc/DR-TANet)]
- **TransCD**: scene change detection via transformer-based architecture, `Optics Express`. [[Paper](https://opg.optica.org/directpdfaccess/fda6838b-0a3c-4d82-a19134fd323dc782_465513/oe-29-25-41409.pdf?da=1&id=465513&seq=0&mobile=no)]
- ![paper](https://img.shields.io/badge/Dataset-red) City-scale Scene Change Detection using Point Clouds, `ICRA`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561855&casa_token=V0SdOlEWYKoAAAAA:fkwwnUHlBk6wejqcW7tRmgxLsOOU7OByZ2t_wh0DwWEVbCn84A2DFF8Hs76fROtWLj4svxPG2g)] [[Code](https://github.com/yewzijian/ChangeDet)] [[Website](https://yewzijian.github.io/ChangeDet/)]
  - Keyword: Point clonds change detection;
- Self-Supervised Pretraining for Scene Change Detection, `NeurIPS`. [[Paper](https://ml4ad.github.io/files/papers2021/Self-Supervised%20Pretraining%20for%20Scene%20Change%20Detection.pdf)] [[Code](https://github.com/NeurAI-Lab/D-SSCD)]
- 3DCD: Scene Independent End-to-End Spatiotemporal Feature Learning Framework for Change Detection in Unseen Videos, `TIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9263106)]
- Change Detection Using Weighted Features for Image-Based Localization, `RAS`. [[Paper](https://erik-derner.github.io/research/files/derner2021change.pdf)]
- Robust change detection based on neural descriptor fields, `IROS`. [[Paper](https://arxiv.org/pdf/2208.01014)]

#### 2020

- ![paper](https://img.shields.io/badge/Dataset-red) Weakly Supervised Silhouette-based Semantic Scene Change Detection, `ICRA`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9196985)] [[Code](https://github.com/kensakurada/sscdnet)] [[Website](https://kensakurada.github.io/pscd/)]
- ![paper](https://img.shields.io/badge/Dataset-red) Epipolar-Guided Deep Object Matching for Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/2007.15540)]
- Change detection with absolute difference of multiscale deep features, Neurocomputing. [[Paper](https://www.sciencedirect.com/science/article/pii/S092523122031290X?casa_token=F5jSf8dXbhcAAAAA:3qghl0AIAYi9AnIER0wis9CMMPlwm5FJoCu5i0Z7VvcSHYkLyKqK-DrJAUGFNhVuGpkKMC2kpJg)]
- **CDNet++**: Improved Change Detection with Deep Neural Network Feature Correlation, `IJCNN`. [[Paper](https://ieeexplore.ieee.org/abstract/document/9207306)]
- Change detection in images using shape-aware siamese convolutional network, `Engineering Applications of Artificial Intelligence`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0952197620301950)]
- Robust and Efficient Object Change Detection by Combining Global Semantic Information and Local Geometric Verification, `IROS`. [[Paper](https://ras.papercept.net/images/temp/IROS/files/1295.pdf)]
  - Keyword: 3D change detection;

#### 2019

- Change detection via graph matching and multi-view geometric constraints, `ICIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8803527&casa_token=tTgScSxhgQgAAAAA:wX7wNAr0wXaYXlt1z8VUpdvNGDdCeCAMVjq3iVbl_TGfNMnzJNDMOIAbpj1hUaucLLtqT9BQGA)]
- **Mask-CDNet**: A mask based pixel change detection network, `Neurocomputing`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231219313979?casa_token=ScCOPDmjskwAAAAA:hr9fF1C6bVEyI9SpQikcLHNIbhjMoFdRNvsNYofkZYu6871gg6Ue2bizD-GhsMepQc1C6IXD6d0)]
- Street-view Change Detection via Siamese Encoder-decoder Structured Convolutional Neural Networks, `VISIGRAPP`. [[Paper](https://www.scitepress.org/Papers/2019/74079/74079.pdf)]

#### 2018

- ![paper](https://img.shields.io/badge/Dataset-red) Street-view change detection with deconvolutional networks, `Autonomous Robots`. [[Paper](https://link.springer.com/article/10.1007/s10514-018-9734-5)]
- Learning to Measure Changes: Fully Convolutional Siamese Metric Networks for Scene Change Detection, `arXiv`. [[Paper](https://arxiv.org/pdf/1810.09111)] [[Code](https://github.com/gmayday1997/SceneChangeDet)]
- **ChangeNet**: A Deep Learning Architecture for Visual Change Detection, `ECCVw`. [[Paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Varghese_ChangeNet_A_Deep_Learning_Architecture_for_Visual_Change_Detection_ECCVW_2018_paper.pdf)] [[Code](https://github.com/leonardoaraujosantos/ChangeNet)]
- **MFCNET**: End-to-end approach for change detection in images, `ICIP`. [[Paper](https://ieeexplore.ieee.org/abstract/document/8451392)]

#### 2017

- ![paper](https://img.shields.io/badge/Dataset-red) TSDF-based Change Detection for Consistent Long-Term Dense Reconstruction and Dynamic Object Discovery, `ICRA`. [[Paper](https://ieeexplore.ieee.org/abstract/document/7989614)] [[Code](https://github.com/ethz-asl/change_detection_ds)]
  - Keyword: 3D change detection;

#### 2014

- ![paper](https://img.shields.io/badge/Dataset-red) **CDnet 2014**: An expanded change detection benchmark dataset, `CVPRw`. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf)] [[Website](http://www.changedetection.net/)] 

#### 2012

- ![paper](https://img.shields.io/badge/Dataset-red) **Changedetection. net**: A new change detection benchmark dataset, `IEEE computer society conference on computer vision and pattern recognition workshops`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6238919)] [[Website](http://www.changedetection.net/)] 

## Application

#### Anomaly Detection

- **UMAD**: University of Macau Anomaly Detection Benchmark Dataset, `IROS`, *2024*. [[Paper]()] [[Dataset](https://github.com/IMRL/UMAD)] [[Website](https://doongli.github.io/umad/)]
- Self-Calibrating Anomaly and Change Detection for Autonomous Inspection Robots, `IRC`, *2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10023636)]
- Survey on video anomaly detection in dynamic scenes with moving cameras, `Artificial Intelligence Review`, *2023*. [[Paper](https://arxiv.org/pdf/2308.07050)]

#### Map Update

- **ExelMap**: Explainable Element-based HD-Map Change Detection and Update, `arXiv`, *2024*. [[Paper](https://arxiv.org/pdf/2409.10178?)]
- **Khronos**: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments, `arXiv`, *2024*. [[Paper](https://arxiv.org/pdf/2402.13817v1)]
- **Lifelong Change Detection**: Continuous Domain Adaptation for Small Object Change Detection in Everyday Robot Navigation, `MVA`, *2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10215686)]
- **POCD**: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes, `RSS`, *2022*. [[Paper](https://arxiv.org/pdf/2205.01202)] [[Dataset](https://github.com/Viky397/TorWICDataset)]
- Probabilistic Object-Level Change Detection for Mapping Semi-Static Scenes, `PhD thesis`, *2022*. [[Paper](https://www.proquest.com/docview/2743543161?pq-origsite=gscholar&fromopenview=true&sourcetype=Dissertations%20&%20Theses)]
- PlaneSDF-Based Change Detection for Long-Term Dense Mapping, `RAL`, *2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9832467)]
- Cumulative Evidence for Scene Change Detection and Local Map Updates, `AIVR`, *2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10024483)]
- Did it change? Learning to Detect Point-of-Interest Changes for Proactive Map Updates, `CVPR`, *2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Revaud_Did_It_Change_Learning_to_Detect_Point-Of-Interest_Changes_for_Proactive_CVPR_2019_paper.pdf)]
- Change Detection and Model Update Framework for Accurate Long-Term Localization, `IROSw`, *2024*. [[Paper](https://hal.science/hal-04733246/document)]
- LT-Gaussian: Long-Term Map Update Using 3D Gaussian Splatting for Autonomous Driving, `arXiv`, *2025*. [[Paper](https://arxiv.org/pdf/2508.01704)]
- Change Detection and Model Update from Limited Query Data for Accurate Robot Localization, `arXiv`, *2025*. [[Paper](https://hal.science/hal-05305364v1/file/paper.pdf)]


#### Localization

- Change detection using weighted features for image-based localization, `RAS`, 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0921889020305169)]

#### Object Rearrangement

- **3DGS-CD**: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement, `RAL`, 2025. [[Paper](https://arxiv.org/pdf/2411.03706)] [[Code](https://github.com/520xyxyzq/3DGS-CD)]

## Citation

This project is part of [UMAD](https://github.com/IMRL/UMAD). If you find this work useful, please consider citing the paper:

```
@inproceedings{li2024umad,
  title={UMAD: University of Macau Anomaly Detection Benchmark Dataset},
  author={Li, Dong and Chen, Lineng and Xu, Cheng-Zhong and Kong, Hui},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5836--5843},
  year={2024},
  organization={IEEE}
}
```
















