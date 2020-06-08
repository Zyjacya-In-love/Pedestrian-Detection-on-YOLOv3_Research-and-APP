"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

def pil_to_base64(img_pil):
    """
    Convert numpy image (RGB) to base64 string
    """
    # img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    img = img_pil
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


from PIL import Image
from io import BytesIO
import base64


def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())




import cv2

def isOpencvImg(img):
    return isinstance(img, np.ndarray)

def PIL2Opencv(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img

def Opencv2PIL(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img
