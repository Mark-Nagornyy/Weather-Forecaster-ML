from PIL import Image
import requests
from io import BytesIO
import streamlit as st
@st.cache_data
def load_and_crop(url, aspect_ratio=(3, 4)):
    """
    Downloads an image and center-crops it to the given aspect ratio.

    aspect_ratio: (width, height)
    """
    img = Image.open(BytesIO(requests.get(url).content))
    w, h = img.size
    target_w = int(h * aspect_ratio[0] / aspect_ratio[1])
    left = (w - target_w) // 2
    right = left + target_w
    return img.crop((left, 0, right, h))