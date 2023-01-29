# This is a separate file for threading reasons
from moviepy.editor import VideoFileClip
from wrapt_timeout_decorator import *
import requests
import os
import numpy as np
from PIL import Image

@timeout(5)
def internal_download_image(url, category):
    filename = url.split('/')[-1].split('?')[0]
    print(f"Downloading {filename} from {url}")
    r = requests.get(url, allow_redirects=True)

    if not os.path.isdir(os.path.join("temp", str(category))):
        os.makedirs(os.path.join("temp", str(category)))

    file_path = os.path.join('temp',str(category), "img_" + filename)
    open(file_path, 'wb').write(r.content)

    clip = VideoFileClip(file_path) 
    clip.iter_frames()

    if not os.path.isdir(os.path.join("output", str(category))):
        os.makedirs(os.path.join("output", str(category)))

    i = 0
    for frame in clip.iter_frames():
        if i > 500: #break in case things get goofy, we don't need this much from one gif
            break
        if i % 5 == 0:
            Image.fromarray(frame).resize((256, 256), Image.LANCZOS).save(os.path.join('output',str(category), filename + f"_{i}" + ".png"))
        i += 1

    clip.close()

    os.remove(file_path)

    return file_path