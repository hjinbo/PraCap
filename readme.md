## Implementation for PraCap

**Read this in other languages: [中文](readme_zh.md).**

### Data

Download the dataset JSON files for COCO and Flickr30k that have been pre-split according to the Karparthy split. Create folders with the corresponding dataset names, each containing 'images' and 'text' subfolders. Place the downloaded JSON files in the 'text' subfolder. Put the two dataset folders in the 'data' directory at the root path.


```
├─data
│  ├─coco
│  │  ├─images
│  │  └─text
│  ├─flickr30k
│  │  ├─images
│  │  └─text
```

### Preprocess

Start by running 'dataset_split.py' in the 'preprocess' directory to obtain the training set, validation set, and test set. After completion, execute 'text_features_extraction.py' in the same directory, paying attention to the parameters of these two files. This will generate .pkl files for the corresponding datasets in the 'preprocess_out' directory.

Create a folder named "others" in the root directory, and then run the method get_support_memory from utils.py.

### Training

Run 'train_cpac_simtexts.py' and pay attention to the parameter specifying the dataset required for training.


### Inference

First, save the downloaded images in the directory data/<dataset name>/images/. Then, run 'image_features_extraction.py' in the 'preprocess' folder to obtain image features. After completion, run 'eval_cpac_simtexts.py' to obtain scores for various evaluation metrics. Pay attention to the parameters when running these scripts.
