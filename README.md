<div align="center">
<h1>
  labelme2coco
</h1>

<a href="https://pepy.tech/project/labelme2coco"><img src="https://pepy.tech/badge/labelme2coco" alt="downloads"></a>
<a href="https://badge.fury.io/py/labelme2coco"><img src="https://badge.fury.io/py/labelme2coco.svg" alt="pypi version"></a>
<a href="https://github.com/fcakyon/labelme2coco/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/labelme2coco/workflows/CI/badge.svg" alt="ci"></a>
<a href="https://twitter.com/fcakyon"><img src="https://img.shields.io/twitter/follow/fcakyon?color=blue&logo=twitter&style=flat" alt="fcakyon twitter">

<h4>
  A lightweight package for converting your <a href="https://github.com/wkentaro/labelme">labelme</a> annotations into COCO object detection format.
</h4>

<h4>
    <img width="700" alt="teaser" src="https://user-images.githubusercontent.com/34196005/148746639-9a7b9c08-2156-42ca-abae-a4e6aad095dd.gif">
</h4>
</div>

## Convert LabelMe annotations to COCO format in one step
[labelme](https://github.com/wkentaro/labelme) is a widely used is a graphical image annotation tool that supports classification, segmentation, instance segmentation and object detection formats.
However, widely used frameworks/models such as Yolact/Solo, Detectron, MMDetection etc. requires COCO formatted annotations.

You can use this package to convert labelme annotations to COCO format.

## Getting started
### Installation
```
pip install -U labelme2coco
```

### Basic Usage

```python
labelme2coco path/to/labelme/dir
```

```python
labelme2coco path/to/labelme/dir --train_split_rate 0.85
```

### Advanced Usage

```python
# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "tests/data/labelme_annot"

# set export dir
export_dir = "tests/data/"

# set train split rate
train_split_rate = 0.85

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir, train_split_rate)
```

```python
# import functions
from labelme2coco import get_coco_from_labelme_folder, save_json

# set labelme training data directory
labelme_train_folder = "tests/data/labelme_annot"

# set labelme validation data directory
labelme_val_folder = "tests/data/labelme_annot"

# set path for coco json to be saved
export_dir = "tests/data/"

# create train coco object
train_coco = get_coco_from_labelme_folder(labelme_train_folder)

# export train coco json
save_json(train_coco.json, export_dir+"train.json")

# create val coco object
val_coco = get_coco_from_labelme_folder(labelme_val_folder, coco_category_list=train_coco.json_categories)

# export val coco json
save_json(val_coco.json, export_dir+"val.json")
```
