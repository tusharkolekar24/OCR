import os
import secrets
import pandas as pd
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_file, session, url_for)

from werkzeug.utils import secure_filename

from datetime import datetime, timedelta
from src.utility import get_model_info
from src.image_model import *

current_date = str(datetime.now()).split(" ")[0]
date_obj = datetime.strptime(current_date, "%Y-%m-%d")

one_year_ago = date_obj + timedelta(days=1)  # 365, 450
current_date_info = one_year_ago.strftime("%Y-%m-%d")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(24))


# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# metainfo
metainfo = dict()
metainfo['type_model']    = ['CNN','VGG16','Xception']
metainfo['input_content']      = '' #'Latest Techqnology used in AI & ML'
metainfo['generated_response'] = ''
metainfo['image_path']         = ''
metainfo['content_info']       = ''
metainfo['filename']           = ''
metainfo['username']           = ''
metainfo['image_metainfo']     = {'image_path':'','process_path':'','extracted_info':'','type_model':''}
# User data for Demonstration
USERS               = {"admin": "1234"}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if USERS.get(username) == password:
            session["username"] = username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials, please try again.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("home_form_data", None)  # Clear home form data from the session
    session.pop("page_form_data", None)  # Clear page form data from the session
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# Define a route for the sectoral page
@app.route('/')
def home():
    if "username" in session:
        content_info = {'input_content'      : metainfo['input_content'],     
                        'generated_response' : metainfo['generated_response']
                        }
        
        metainfo['content_info'] = content_info
        filename                 = metainfo['filename']  
        image_path               = metainfo['image_path']
        metainfo['username']     = session["username"]
        
        if metainfo['image_path']!='':
            print('Image Found')

        return render_template(
            "home.html",
            username     = metainfo['username'],
            form_data    = metainfo,
            current_date = current_date,
            content_info = content_info,
            image_path   = metainfo['image_metainfo'],
            filename     = filename
        )
    
    return redirect(url_for("login"))

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():

    content_info = {'input_content'          : metainfo['input_content'],     
                        'generated_response' : metainfo['generated_response']
                    }

    if 'fileUpload' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['fileUpload']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        filename     = secure_filename(file.filename)
        file_path    = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        process_path = os.path.join(app.config['UPLOAD_FOLDER'],filename.split(".")[0]+'_process'+'.'+filename.split(".")[1])
        
        # print(process_path,'process_path')
        # print(file_path,'file_path')

        # Read image from the uploaded file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            flash('Invalid image file')
            return redirect(url_for('home'))

        # Resize the image to desired dimensions (e.g., 450x250)
        resized_image = cv2.resize(image, (450, 250), interpolation=cv2.INTER_LINEAR)

        # Save the resized image to the upload folder
        cv2.imwrite(file_path, resized_image)

        mapping, Upgraded_Image, Image_ML = images_preprocessing(image)
        
        print(metainfo['type_model'][0],'Model Selected Toperform Predictions')

        loaded_model, encoder_json = get_model_info(metainfo['type_model'][0])

        transform_image, extracted_text = extract_data_from_images(mergeset        = mapping,
                                                                   Images_Rotation = Upgraded_Image,
                                                                   ml_prediction   = Image_ML,
                                                                   loaded_model    = loaded_model,
                                                                   encoder_json    = encoder_json,
                                                                   model_name      = metainfo['type_model'][0])
        
        # Save the resized image to the upload folder
        cv2.imwrite(process_path, transform_image)

 
        # Send the path of the uploaded image
        metainfo['image_path'] = file_path
        metainfo['filename']   = filename

        # metainfo['image_metainfo']     = {'image_path':'','process_path':'','extracted_info':''}
        
        metainfo['image_metainfo']['image_path']     = file_path
        metainfo['image_metainfo']['process_path']   = process_path
        metainfo['image_metainfo']['extracted_info'] = extracted_text
        metainfo['image_metainfo']['type_model']     = metainfo['type_model'][0]

        # metainfo['image_metainfo']     =
        # metainfo['username']   = session["username"]

        return render_template(
            "home.html",
            username     = metainfo['username'],
            form_data    = metainfo,
            current_date = current_date,
            content_info = content_info,
            image_path   = metainfo['image_metainfo'],
            filename     = filename
        )
    
    return redirect(url_for("login"))


@app.route("/submit_home_form", methods=["POST"])
def submit_home_form():
    # Retrieve form data from home.html
    # type_model,temp_info
    content_info = {'input_content'          : metainfo['input_content'],     
                        'generated_response' : metainfo['generated_response']
                    }
    model_type_info  = request.form.get('type_model')
    # temp_info        = request.form.get('temp_info')
    
    type_model_list = [model_type_info]
    
    for cols in ['CNN','VGG16','Xception']:
        if cols not in type_model_list:
            type_model_list.append(cols)

    metainfo['type_model']       = type_model_list

    metainfo['image_metainfo']['type_model']       = type_model_list[0]
    # metainfo['temp_infos']       = temp_info_list
    # flash("Form submitted successfully for Home page!", "success")

    return render_template(
            "home.html",
            username     = metainfo['username'],
            form_data    = metainfo,
            current_date = current_date,
            content_info = content_info,
            image_path   = metainfo['image_metainfo'],
            filename     = metainfo['filename'] 
        )


@app.route("/topic_info_update", methods=["POST"])
def topic_info_update():
    # Retrieve form data from home.html
    flash("Form submitted successfully for Home page!", "success")
    return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(debug=True)