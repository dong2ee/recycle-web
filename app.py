import io
from operator import truediv
import os
import json
from PIL import Image

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def get_model_names():
    
    model_names = []
    for f  in os.listdir():
        if f.endswith(".pt"):
            model_names.append(f.split(".")[0])
    return model_names



def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")
    
global_model_name = find_model()
model =torch.hub.load("WongKinYiu/yolov7", 'custom',global_model_name)

model.eval()

def get_prediction(img_bytes, model_name):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    model =torch.hub.load("WongKinYiu/yolov7", 'custom', model_name+".pt")
    model.eval()
    results = model(imgs, size=640)  
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        model_name = request.form["model_name"]
        if not file:
            return
            
        img_bytes = file.read()
        results = get_prediction(img_bytes, model_name=model_name)
        results.save(save_dir='static/result')
        filename = 'image0.jpg'
        
        return render_template('result.html',result_image = filename,model_name = model_name)
    return render_template('index.html', model_names = get_model_names())

if __name__=="__main__":
    app.run('0,0,0,0', port=5000, debug=True)
