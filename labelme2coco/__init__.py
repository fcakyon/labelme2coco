from __future__ import absolute_import

__version__ = "0.1.2"

from labelme2coco.labelme2coco import labelme2coco

def converter():
    return labelme2coco()

def convert(labelme_folder: str, save_json_path: str):
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        save_json_path: oath for coco json to be saved
    """
    labelme2coco().convert(labelme_folder, save_json_path)
