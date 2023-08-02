# python -m tools.draw_gt_box
import os
import json

from glob import glob
from PIL import Image, ImageDraw
from config.train_test_cfg import cfg

if __name__ == "__main__":
    img_file_list = glob(os.path.join(cfg.test_imgs_dir, "*.jpg"))

    if not os.path.isdir(cfg.result_imgs_dir):
            os.mkdir(cfg.result_imgs_dir)

    for img_file in img_file_list:
        json_file = img_file.replace("jpg", "json")

        img = Image.open(img_file)
        with open(json_file, "r") as f:
            json_data = json.loads(f.read())

        for key in json_data.keys():
            lbl = json_data[key]["category"]
            x1 = json_data[key]["x1"]
            y1 = json_data[key]["y1"]
            x2 = json_data[key]["x2"]
            y2 = json_data[key]["y2"]
            draw_rectangle = ImageDraw.ImageDraw(img)
            draw_rectangle.rectangle(((x1, y1), (x2, y2)), fill=None, outline=cfg.obj_color[lbl], width=2)

        img.save(os.path.join(cfg.result_imgs_dir, img_file.split(os.sep)[-1]))