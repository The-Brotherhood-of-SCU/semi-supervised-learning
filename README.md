# 2024人工智能引论大作业：半监督服装分类

# Overview
本项目是2024人工智能引论的大作业，主要是使用半监督学习方法对服装进行分类。


# Files
- `final_presentation.pdf`: 实验报告
- `fashion_dataset.py` : 数据集加载&增强
- `module.py`: 模型定义(包含3个模型，分别对应以下3个文件训练)
- `main.py`: 训练代码(CNN模型)
- `train_try1.py`: 还是别看这个代码了，写的太丢人了
- `VAE.py`: 训练代码(VAE模型（但好像这个不是标准的VAE）)

其中VAE模型表现最好，Accuracy达到了班级最高。更详细的内容可以参看[实验报告](final_presentation.pdf)`report.pdf`。

# VAE模型简介

执行多任务学习。

```
            ->Decoder      =>图像重建
Encoder->低维表示
            ->Classifier   =>分类
```

对于分类任务，使用交叉熵损失函数。
对于图像重建任务，使用均方误差损失函数。

对于有标记的样本，两者产生的损失函数均可训练。
对于无标记的样本，仅使用图像重建任务的损失函数进行训练。