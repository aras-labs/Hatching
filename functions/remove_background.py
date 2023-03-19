import pixellib
from pixellib.tune_bg import alter_bg

def remove_bg(file_name):
    change_bg = alter_bg(model_type = "pb")
    change_bg.load_pascalvoc_model("models/xception_pascalvoc.pb")
    img = change_bg.color_bg(file_name,  colors = (255,255,255), detect ="person")
    return img