## PraCap实现

**其他语言版本: [English](readme.md).**

### 数据

分别下载已经按照karparthy split划分好coco和flickr30k的数据集json文件，以对应数据集名创建文件夹，其中包含images和text子文件夹，将下载好的json文件放在text中，将两个数据集文件夹置于根路径的data目录下。

```
├─data
│  ├─coco
│  │  ├─images
│  │  └─text
│  ├─flickr30k
│  │  ├─images
│  │  └─text
```

### 预处理

先运行preprocess目录下的dataset_split.py获得训练集，验证集和测试集，执行完毕后运行同目录下的text_features_extraction.py，运行时需要注意这两个文件的参数。此时，将在preprocess_out目录下生成对应数据集的pkl文件。

在根目录下创建others/文件夹，然后运行utils.py中get_support_memory方法。

### 训练

运行train_cpac_simtexts.py，注意所需训练的数据集这一参数。

### 预测

先将下载好的图像保存在data/<对应数据集目录>/images/下，运行preprocess中的image_features_extraction.py，获取图像特征，执行完毕后运行eval_cpac_simtexts.py，即可获得各评价指标的分数。同样注意运行时的参数。