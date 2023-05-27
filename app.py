from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model = load_model("cervical_cancer.h5")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/',methods=['POST','GET'])
def cancerPrediction():
    if request.method == "POST":
        image = request.files["filename"]
        file_path = os.path.join (app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(file_path)
    classes_dir = ["Dyskeratotic","Koilocytotic","Metaplastic","Parabasal","Superficial-Intermediate"]
    # Loading Image
    img = tf.keras.utils.load_img(file_path, target_size=(64,64))
    # Normalizing Image
    norm_img =  tf.keras.utils.img_to_array(img)/255
    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])
    # Getting Predictions
    pred = np.argmax(model.predict(input_arr_img))
    # Printing Model Prediction
    prediction=classes_dir[pred]
    dic = {"Dyskeratotic":"Cervical dysplasia is the abnormal growth of cells on the surface of the cervix. Considered a precancerous condition, it is caused by a sexually transmitted infection with a common virus, the Human Papillomavirus (HPV).",
           "Koilocytotic":"A koilocyte is a squamous epithelial cell that has undergone a number of structural changes, which occur as a result of infection of the cell by human papillomavirus (HPV). Identification of these cells by pathologists can be useful in diagnosing various HPV-associated lesions.",
           "Metaplastic":"Squamous metaplasia can affect any part of your epithelium. It most commonly develops in the mucus-making cells that line your endocervix, a part of the female reproductive system. This passageway inside your cervix connects your uterus and vagina",
           "Parabasal":"Parabasals are an uncommon finding on Pap smears of women with estrogen production or replacement hormone. These cells are often seen in patients who lack estrogen, including those who are premenstrual, post partum, taking estrogen-restricting hormones, or postmenopausal.",
           "Superficial-Intermediate":"Superficial squamous cells are seen in abundance during the late proliferative and ovulatory phases of the menstrual cycle. At these points, estrogen is at its peak."}
    # print("dic['Dyskeratotic']: ", dic['Dyskeratotic'])
    print(prediction)
    print(classes_dir[pred])
    print(dic[prediction])
    desc = dic[prediction]
    file=file_path.replace("static/","")
    file=file.replace("\\","/")
    print(file)
    return render_template('index.html',prediction=prediction,pred = desc,files=file)



if __name__ == "__main__":
    app.run(debug = True)