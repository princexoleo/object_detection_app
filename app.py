"""
Main file for Flask web application.
Simple application which have image upload features via web form,
and view the inference results on the image in the browser.

"""

# Libraires import

import argparse
import io
import os 
from PIL import Image 

# PyTorch Libraries 
import torch

# Flask Libraries 
from flask import Flask, render_template, request, redirect

# Initialize the flask project with the app name
app_name = "Object Detection App"
app = Flask(__name__)


# Initialize the default route for flask application
@app.route("/", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if not file:
            return
        img_bytes  = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        # after reading the image then needs to 
        # pass through the model
        results = model(img, size=640)
        results.display(save=True, save_dir="object_detection_app/static")
        return redirect("object_detection_app/static/image0.jpg")
    
    return render_template("index.html")


# When this file is called
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    # load models from torch.hub
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code

    # Custom Vehicle Classification Model load
    # if os.path.isfile("models/best.pt"):
    #     model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')  # default
    # else:
    #     print("ModelNotFound")
    
    model.eval()
    app.run(
        host='localhost',
        port=args.port,
        debug=True,
        )