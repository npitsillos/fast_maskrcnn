# fast_maskrcnn

A [mask_rcnn](https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn) model implementation running on a [FastAPI](https://github.com/tiangolo/fastapi) endpoint using [streamlit](https://github.com/streamlit/streamlit) to input an image and display the results.

# Usage
Download the repository and install [poetry](https://github.com/python-poetry/poetry).  Then open up two terminals and run the following commands for FastAPI and streamlit respectively:

```
poetry run uvicorn fast_maskrcnn.app:app
poetry streamlit fast_maskrcnn/main.py
```