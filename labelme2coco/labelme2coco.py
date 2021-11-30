import os
import json
import PIL.Image
import PIL.ImageDraw
import numpy as np
from labelme2coco.utils import create_dir, list_jsons_recursively
from labelme2coco.image_utils import read_image_shape_as_dict


class labelme2coco(object):
    def __init__(self):
        # Retains categories for different datasets. e.g. train and validation set
        self.categories = []
        self.label = []

    def convert(self, labelme_folder='', save_json_path='./new.json'):
        """
        Args:
            labelme_folder: folder that contains labelme annotations and image files
            save_json_path: path for coco json to be saved
        """
        self.save_json_path = save_json_path
        self.images = []
        self.annotations = []
        self.annID = 1
        self.height = 0
        self.width = 0

        # create save dir
        save_json_dir = os.path.dirname(save_json_path)
        create_dir(save_json_dir)

        # get json list
        _, labelme_json = list_jsons_recursively(labelme_folder)
        self.labelme_json = labelme_json

        self.save_json()

    def data_transfer(self):
        for num, json_path in enumerate(self.labelme_json):
            with open(json_path, 'r') as fp:
                # load json
                data = json.load(fp)
#                (prefix, res) = os.path.split(json_path)
#                (file_name, extension) = os.path.splitext(res)
                self.images.append(self.image(data, num, json_path))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.category(label))
                        self.label.append(label)
                    points = shapes['points']
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num, json_path):
        image = {}
        # get image path
        _, img_extension = os.path.splitext(data["imagePath"])
        image_path = json_path.replace(".json", img_extension)
        img_shape = read_image_shape_as_dict(image_path)
        height, width = img_shape['height'], img_shape['width']

        image['height'] = height
        image['width'] = width
        image['id'] = int(num + 1)
        image['file_name'] = image_path

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category['supercategory'] = label
        category['id'] = int(len(self.label) + 1)
        category['name'] = label

        return category

    def annotation(self, points, label, num):
        annotation = {}
        annotation['iscrowd'] = 0
        annotation['image_id'] = int(num + 1)

        annotation['bbox'] = list(map(float, self.getbbox(points)))

        # coarsely from bbox to segmentation
        x = annotation['bbox'][0]
        y = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = int(self.annID)
        # add area info
        annotation['area'] = self.height * self.width  # the area is not used for detection
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
            # if label[1]==categorie['name']:
            #     return categorie['id']
        return -1

    def getbbox(self,points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)
        # cv2.fillPoly(img, [np.asarray(points)], 1)
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]  # [x1,y1,w,h] for coco box format

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()

        json.dump(self.data_coco, open(self.save_json_path, 'w', encoding='utf-8'), indent=4, separators=(',', ': '), cls=MyEncoder)


# type check when save json files
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    labelme_folder = "tests/data/labelme_annot"
    save_json_path = "tests/data/test_coco.json"
    labelme2coco(labelme_folder, save_json_path)
