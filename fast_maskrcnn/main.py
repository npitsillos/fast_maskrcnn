import streamlit as st
import base64
import json
import requests
import numpy as np

from utils import FILE_TYPES, IP_ADDRESS, draw_bounding_box, bytes_to_PIL_image

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Mask RCNN with FastAPI")
file_buffer = st.file_uploader("Please upload an image", type=FILE_TYPES)

if file_buffer is not None:
    img_bytes = file_buffer.read()
    st.image(img_bytes, caption="Test image")

if st.button("Detect Objects"):
    if file_buffer is None:
        st.write("No image uploaded...")
    else:
        img = bytes_to_PIL_image(img_bytes)
        img_bytes = base64.b64encode(img_bytes)
        img_bytes = img_bytes.decode("utf-8")
        payload = json.dumps({"img_bytes": img_bytes})
        res = requests.put(IP_ADDRESS, payload)
        json_object = res.json()
        img = np.asarray(img)
        img = draw_bounding_box(img, json_object["boxes"], json_object["classes"])
        st.image(img, caption="Processed image")