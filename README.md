# **论文复现：CycleMLP: A MLP-like Architecture for Dense Prediction**
* 用Paddle复现[CycleMLP: A MLP-like Architecture for Dense Prediction](https://arxiv.org/pdf/2107.10224v1.pdf)论文

* **精度对比:**

| Model| paddlepaddle | pytorch | diff |
| -------- | -------- | -------- | -------- |
| CycleMLP-B1 | 78.794 | 78.9 | -0.106 |
| CycleMLP-B2 | 81.508 |	81.6 | -0.092 |
| CycleMLP-B3 | 82.274 |	82.4 | -0.126 |
| CycleMLP-B4 | 82.962 |	83.0 | -0.038 |
| CycleMLP-B5 | 83.25  |	83.2 | +0.05 |
## 一、Model Zoo
[paddle模型参数压缩包下载地址（CycleMLP-B1/2/3/4/5.pdparams）](https://aistudio.baidu.com/aistudio/datasetdetail/107267/0)

## 二、复现细节（基于PaddleClas）：
### 1. 超参数设计

| 超参数 | 设置值 |
| -------- | -------- |
| momentum     | 0.9     |
| weight decay     | 5x10^-2     |
| learning rate     | 1x10^-3     |
| epochs     | 300     |
| batch size     | 1024     |

* **注：上述参数设置是在  8 Tesla V100 GPUs，在单卡上 batch size 减少倍数与 learning rate 减少倍数相同。**

### 2.数据增强
* 部分数据增强来自于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

* Mixup 混合数据增强
> 它可以将不同类之间的图像进行混合，从而扩充训练数据集。

### 3.知识蒸馏
* [李宏毅机器学习进阶-知识萃取](https://aistudio.baidu.com/aistudio/course/introduce/1979)
* 教师模型来自于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
* **注：pip安装PaddleClas可能会卡在open-cv那里，可以下载压缩包，教师模型参数也需要下载在本地。**

## 三、数据集
### 1. [下载地址](https://image-net.org/download.php)
### 2. 组织方式
* 一个文件夹下放置图片，例如：

![](https://ai-studio-static-online.cdn.bcebos.com/c34687f59ba94e428e4e5fba8beb5cba1a8fbfca35094eebbd63168379e9c9db)

* 一个.txt文件，内含图片名，标签（以空格隔开），例如：

![](https://ai-studio-static-online.cdn.bcebos.com/1dd0e45052f34a12b58ca060bf9d237e33de941a155542e5ac2034aa9c65353a)


### 3.图片类别
* map_to_1000.txt 文件内含类别映射，需要关注图片文件名前面几个字段。


## 四、[AI Studio 演示](https://aistudio.baidu.com/aistudio/projectdetail/2343936)
### 1. 解压测试数据


```python
!mkdir /home/aistudio/data/tar
!mkdir /home/aistudio/data/train_data00
!cd /home/aistudio/data/tar/;cat /home/aistudio/data/data9244/train.tar.00 | tar -x
!cd /home/aistudio/data/train_data00;ls /home/aistudio/data/tar/*.tar | xargs -n1 tar xf
#显示work/train中图片数量
!find /home/aistudio/data/train_data00 -type f | wc -l
!rm -rf /home/aistudio/data/tar

!mkdir /home/aistudio/data/val_data
!mkdir /home/aistudio/data/pretrained_pdparams
!tar -xf /home/aistudio/data/data9244/ILSVRC2012_img_val.tar -C /home/aistudio/data/val_data
!unzip -oq data/data107267/CycleMLP_pretrained.zip -d /home/aistudio/data/pretrained_pdparams
```

### 2. 测试训练
* 单机单卡


```python
%cd CycleMLP/
!python train.py --train-data-dir /home/aistudio/data/train_data00 --train-txt-path /home/aistudio/train00.txt \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt \
                --epochs 2 --batch-size 256 --lr 1e-8 --distillation-type soft \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B1.pdparams \
                --teacher-pretrained /home/aistudio/RegNetX_4GF_pretrained.pdparams
```

* 自动混合精度训练


```python
%cd CycleMLP/
!python train.py --train-data-dir /home/aistudio/data/train_data00 --train-txt-path /home/aistudio/train00.txt \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt \
                --epochs 2 --batch-size 256 --lr 1e-8 --distillation-type soft \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B1.pdparams \
                --teacher-pretrained /home/aistudio/RegNetX_4GF_pretrained.pdparams \
                --is_amp True
```

* 单机多卡


```python
%cd CycleMLP/
!python -m paddle.distributed.launch \
        train.py --train-data-dir /home/aistudio/data/train_data00 --train-txt-path /home/aistudio/train00.txt \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt \
                --epochs 2 --batch-size 256 --lr 1e-8 --distillation-type soft \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B1.pdparams \
                --teacher-pretrained /home/aistudio/RegNetX_4GF_pretrained.pdparams \
                --is_distributed True
```

### 3. 验证精度对齐
##### ① CycleMLP-B1


```python
%cd CycleMLP/
!python eval.py --model CycleMLP_B1 \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B1.pdparams \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt
```

#### ② CycleMLP_B2



```python
%cd CycleMLP/
!python eval.py --model CycleMLP_B2 \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B2.pdparams \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt
```

####  ③ CycleMLP_B3


```python
%cd CycleMLP/
!python eval.py --model CycleMLP_B3 --batch-size 10 \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B3.pdparams \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt
```

#### ④ CycleMLP_B4


```python
%cd CycleMLP/
!python eval.py --model CycleMLP_B4 --batch-size 10 \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B4.pdparams \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt
```

#### ⑤ CycleMLP_B5


```python
%cd CycleMLP/
!python eval.py --model CycleMLP_B5 --batch-size 10 \
                --model-pretrained /home/aistudio/data/pretrained_pdparams/CycleMLP_pretrained/paddle_CycleMLP_B5.pdparams \
                --val-data-dir /home/aistudio/data/val_data --val-txt-path /home/aistudio/val.txt
```

## 五、for seg and det

> * pip install [paddleseg](https://github.com/PaddlePaddle/PaddleSeg)
> * pip install [paddledet](https://github.com/PaddlePaddle/PaddleDetection)
> * cycle_mlp as backbone

## 六、总结
### 1. 本文提出了一个简单的 MLP-like 的架构 CycleMLP，它是视觉识别和密集预测的通用主干，不同于现代 MLP 架构，例如 MLP-Mixer、ResMLP 和 gMLP，其架构与图像大小相关，因此是在目标检测和分割中不可行。
### 2. CycleMLP有两点优点:
#### (a) 它可以处理不同大小的图像。
#### (b) 它利用局部窗口实现了图像大小的线性计算复杂度。

![](https://ai-studio-static-online.cdn.bcebos.com/134db21b8c8f4e6ca2199bb89fe981a065885e24c2eb436b88d79162cf080c00)

### 3. **Motivation of Cycle Fully-Connected Layer (FC)** compared to Channel FC and Spatial FC.
#### (a) 通道FC聚集在空间尺寸为“1”的通道维上。它处理各种输入尺度，但不能学习空间背景。
#### (b) 空间FC在空间维度上具有全局的接受域。但其参数大小是固定的，对图像尺度具有二次计算复杂度。
#### (c) 我们建议的周期全连接层(Cycle FC)具有与信道FC相同的线性复杂性，且接收域比信道FC大通道FC。

![](https://ai-studio-static-online.cdn.bcebos.com/4dc689fc7b7f44479de28bf5ed3c4dcec14a57a4eea44390bd7c5a2cbfff22a5)

### 4. **Comparison of the MLP blocks.**
#### (a) MLPMixer(左)沿着空间维度使用普通MLP进行上下文聚合。当输入尺度变化时，该运算符不能工作，并且对图像大小具有二次计算复杂度。
#### (b) 我们的CycleMLP使用用于空间投影的周期FC，能够处理任意尺度，具有线性复杂度。

### 5. **Pseudo-Kernel.**
#### (a) 将Cycle FC的采样点(橙色块)投影到空间表面上，并将投影面积定义为伪核大小。
![](https://ai-studio-static-online.cdn.bcebos.com/c96f9eb93a9d4a72a72682f9abfd9419a60ead43f231415ba5c712a06633b3d7)

### 6. **CycleMLP 模型结构.**
####  (a) Ei和Li表示扩展比例和重复层数。我们的设计原则是受到ResNet理念的启发，其中通道尺寸增加，而空间分辨率随着层的加深而缩小。
![](https://ai-studio-static-online.cdn.bcebos.com/01e1d8a391d04c7888eded5abfa11502bef7703f6e9b4bc8b136dbbc0ef99237)

### 7.**ImageNet accuracy v.s. model capacity.**
#### (a) 所有模型都在ImageNet-1K[11]上训练，没有额外的数据。CycleMLP超越了现有的类似mlp的模型，如MLP-Mixer， ResMLP， gMLP， S2-MLP和ViP。
![](https://ai-studio-static-online.cdn.bcebos.com/4c3c4a83e481417f85ecc262578ea099a129a86d3fff4f7299815a7a3724488d)

