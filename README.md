# Study on the effectiveness of novel deep learning methods for fault diagnosis

This is a repository for studying the effectiveness of deep learning methods in the domain of **fault diagnosis**. 

This subject contains papers, codes, and data used for discussion in this article.

## Content

 - [Papers of diagnosis](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#papers)
	 - [Survey](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#survey)
	 - [Fault diagnosis method based on CNN](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#Fault-diagnosis-method-based-on-CNN)
	 - [ Fault diagnosis method based on LSTM](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#Fault-diagnosis-method-based-on-lstm)
	 - [Fault diagnosis method with attention mechanism](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#Fault-diagnosis-method-with-attention-mechanism)
 - [code](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#code)
 - [data](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master?tab=readme-ov-file#data)
 
 ## 📝Papers
 
> We list the papers according to the basic framework of the method, which can be roughly divided into **CNN-based**, **LSTM-based** and **attention mechanism**
### Survey
***
 - Are Novel Deep Learning Methods Effective for Fault Diagnosis？[**[IEEE TR 2024]**]()

### Fault diagnosis method based on CNN
***
 - An Efficient Sequential Embedding ConvNet for Rotating Machinery Intelligent Fault Diagnosis (**SECN**) [[**TIM 2023**]](https://ieeexplore.ieee.org/abstract/document/10102489/)
 - Deep residual learning-based fault diagnosis method for rotating machinery (**DSL**) [[**ISA 2019**]](https://www.sciencedirect.com/science/article/abs/pii/S0019057818305202)
 - A Fault Diagnosis Method for Rotating Machinery Based on CNN With Mixed Information (**MIXCNN**) [[**TII 2022**]](https://ieeexplore.ieee.org/abstract/document/9964316)
 - WaveletKernelNet: An Interpretable Deep Neural Network for Industrial Intelligent Diagnosis (**WaveletKernelNet**) [[**TSMCS 2021**]](https://ieeexplore.ieee.org/abstract/document/9328876)
 - Deep Residual Shrinkage Networks for Fault Diagnosis (**RSBU**) [[**TII 2020**]](https://ieeexplore.ieee.org/abstract/document/8850096)
 - Understanding and Learning Discriminant Features based on Multiattention 1DCNN for Wheelset Bearing Fault Diagnosis (**MA1DCNN**) [[**TII 2020**]](https://ieeexplore.ieee.org/abstract/document/8911240)
### Fault diagnosis method based on LSTM
***
 - Gearbox fault diagnosis based on Multi-Scale deep residual learning and
stacked LSTM model (**MDRL-SLSTM**) [[**MEAS 2021**]](https://www.sciencedirect.com/science/article/abs/pii/S0263224121010216)
 - Interpreting network knowledge with attention mechanism for bearing fault diagnosis (**ATTMBIGRU**) [[**ASC 202**]](https://www.sciencedirect.com/science/article/abs/pii/S1568494620307675)
### Fault diagnosis method with attention mechanism
>Please note that the method of introducing the attention mechanism is also based on CNN or LSTM.
***
 - Interpreting network knowledge with attention mechanism for bearing fault diagnosis (**ATTMBIGRU**) [[**ASC 202**]](https://www.sciencedirect.com/science/article/abs/pii/S1568494620307675)
 - Understanding and Learning Discriminant Features based on Multiattention 1DCNN for Wheelset Bearing Fault Diagnosis (**MA1DCNN**) [[**TII 2020**]](https://ieeexplore.ieee.org/abstract/document/8911240)

 ## 🔧Code
 

> The provided code is basically unofficial, but the team reproduced it through the experimental details provided in the above paper

Our code is released at [[**Code link**]](https://github.com/sumyinho/Novel-Models-for-Fault-Diagnosis/tree/master)

You can check the structure framework of different models by looking at `model_name.py` in the `model` file.

If you want to see the details and settings of the code running, you can try to reproduce it by running `exp_model_name.py` in the `experiment` file


## 👜Data

> The three public datasets used in this article are provided, including the original dataset and the selected data in this paper.
>
You first need to create a `dataset` folder in the directory and create separate files for different datasets.
```
.
└── datasets
    └── CWRU
        ├── 97.mat
        │── 98.mat
        │── 106.mat
        │   ...
        │── 162.mat
    └── XJTU
        ├── Bearing2_1
        │   ├── 1.csv
        │   │── 2.csv
        │   ...
        ├── Bearing2_2
        ...
```
|  Index| Year|Dataset name|Component|Original data link|Selected data Link|
|--|--|--|--|--|--|
| 01 | 2015 |CWRU|bearing|[[data link]](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures)|[[data link]](https://pan.quark.cn/s/88cfe9985bc7)|
|02|2016|PU|bearing|[[data link]](https://groups.uni-paderborn.de/kat/BearingDataCenter/)|[[data link]](https://pan.quark.cn/s/88cfe9985bc7)|
|03|2018|XJTU|bearing|[[data link]](https://biaowang.tech/xjtu-sy-bearing-datasets/)|[[data link]](https://pan.quark.cn/s/88cfe9985bc7)|

## :telephone:Contact
If you have any problem, please feel free to contact me.
Name: Chenxian He
Email address: [sumyin.ho@outlook.com](mailto:sumyin.ho@outlook.com)

# 🔎BibTex Citation
If you find this paper and repository useful, please cite our paper. It helps in the recognition and dissemination of our work. Please use the following citation format🤗:
```

```
