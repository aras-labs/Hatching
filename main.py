import os
import json
import cv2
from datetime import datetime

from hatching import hatch
from functions import remove_bg, find_levels, sculpt


def convert(file_name, output_path, params):
    with open("config.json") as jsonfile:
        config = json.load(jsonfile)

    img = remove_bg(file_name)
    img = sculpt(img)
    lvls = tuple(find_levels(img))
    print(lvls)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hatch(img, output_path=output_path, circular=False, 
          image_scale=float(params["image_scale"]), hatch_angle=int(params["hatch_angle"]), board_config=config, 
          hatch_pitch=1.5, levels=lvls, blur_radius=1)


if __name__ == "__main__":
    dt = datetime.today()  
    seconds = dt.timestamp()

    os.mkdir(f"outputs/{seconds}")
    convert("Taghirad.jpg", f"outputs/{seconds}")