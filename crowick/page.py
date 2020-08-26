import functools
import os
import io
import base64
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from PIL import Image
from yolo.main import predict, load_model
YOLO_MODEL_SIZE = (608, 608)

model = load_model(log = True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
TMP_PATH = "crowick/static/tmp"

bp = Blueprint('page', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template("index.html")

@bp.route('/object-detection', methods=["GET", "POST"])
def object_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            return render_template("object-detection.html")
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template("osbject-detection.html")
        if file and allowed_file(file.filename):
            # make sure filename is not malicious
            filename = secure_filename(file.filename)

            # save temporary file
            image_path = os.path.join(TMP_PATH, filename)
            file.save(image_path)

            # process
            image = predict(image_path, model, YOLO_MODEL_SIZE, log=True)

            # load image to memory
            file_object = io.BytesIO()
            image.save(file_object, 'PNG') 
            file_object.seek(0)

            # retrieve and encode loaded image
            image = base64.b64encode(file_object.getvalue())

            os.remove(image_path)
        return render_template("object-detection.html", image=image.decode('ascii'))
    else:
        return render_template("object-detection.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS