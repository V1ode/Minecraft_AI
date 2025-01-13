import flask
import numpy as np
import math
import cv2
from flask import Flask, render_template, request, jsonify, Response
from keras.src.saving import load_model
import tensorflow as tf
from sahi.utils.torch import torch
from tensorflow.keras.preprocessing import image
import os
import io
from PIL import Image


app = Flask(__name__)

menu = [{"name": "Дозорный", "url": "p_yolov5sDetection"},
        {"name": "Передовой ученый", "url": "p_pretrainedClassification"},
        {"name": "Ученый-любитель", "url": "p_myClassification"}
        ]


train_dataset = tf.keras.utils.image_dataset_from_directory(
  "static/media/train/Detection/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(96, 96),
  batch_size=32)

detection_classes = ["Белая овечка", "Житель", "Корова", "Крипер", "Свинья"]

train_dataset = tf.keras.utils.image_dataset_from_directory(
  "static/media/train/Classification/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(96, 96),
  batch_size=32)

classification_classes = sorted(train_dataset.class_names)

classification_classes_rus = {
    "Banner": "Баннер",
    "Bed": "Кровать",
    "Block": "Блок",
    "Candle": "Свеча",
    "Disc": "Диск",
    "Door": "Дверь",
    "Gabion": "Габион",
    "Missile": "Снаряд",
    "Potion": "Зелье",
    "Sign": "Табличка",
    "Slab": "Полублок",
    "Spawn Egg": "Яйцо призыва",
    "Stained glass pane": "Стеклянная панель",
    "Stairs": "Лестница",
    "Trapdoor": "Люк",
}


detection_classes_rus = {
    "white_sheep": "Белая овечка",
    "villager": "Житель",
    "cow": "Корова",
    "creeper": "Крипер",
    "pig": "Свинья"
}

detection_classes_eng = {
    "Белая овечка": "white_sheep",
    "Житель": "villager",
    "Корова": "cow",
    "Крипер": "creeper",
    "Свинья": "pig"
}

detection_class_color = {
    "white_sheep": (0, 238, 255),
    "villager": (68, 0, 255),
    "cow": (255, 0, 0),
    "creeper": (71, 167, 106),
    "pig": (255, 151, 187)
}


grouped_classes = [detection_classes[:len(detection_classes)//2+1], detection_classes[len(detection_classes)//2+1:]]

# Подгружаем предобученную нейронку классификации
pretrained_model = load_model("saved_models/pretrainedModel100.keras")

# Подгружаем собственную нейронку классификации
own_model = load_model("saved_models/myModel500.keras")

# Подгружаем модель YOLOv5s
YOLOv5s_model = torch.hub.load('ultralytics/yolov5', model="custom", path="model/best.pt", force_reload=True)


# Роуты

@app.route("/")
def index():
    return render_template('index.html', menu=menu)


@app.route("/p_yolov5sDetection", methods=['POST', 'GET'])
def f_yolov5sDetection():
    if request.method == 'GET':
        return render_template('YOLO5sDetection.html', title="Дозорный", menu=menu, class_model='', grouped_classes=grouped_classes)
    if request.method == 'POST':
        if (os.path.exists('static/media/detection_blank.png')):
            os.remove('static/media/detection_blank.png')

        img_file = flask.request.files.get('img', '')
        img_file.save('static/media/detection_blank.png')

        confidence = float(flask.request.form["confidence"])
        chosen_classes = flask.request.form.getlist("classes")

        chosen_classes_eng = []
        for classname in chosen_classes:
            chosen_classes_eng.append(detection_classes_eng[classname])

        img = image.load_img('static/media/detection_blank.png', target_size=(640, 640))

        generated = YOLOv5s_model(img)

        filtered_results = generated.pandas().xyxy[0]
        filtered_results = filtered_results[
            (filtered_results['name'].isin(chosen_classes_eng)) & (filtered_results['confidence'] >= confidence)]

        filtered_image = np.array(img)
        for _, row in filtered_results.iterrows():
            label = detection_classes_rus[row['name']]
            conf = row['confidence']
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
            color = detection_class_color[row['name']]
            cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_COMPLEX,
                        0.9, color, 3)

        filtered_image = Image.fromarray(filtered_image)
        filtered_image.save('static/media/detection_result.jpg')

        return render_template('YOLO5sDetectionResult.html', title="Дозорный", menu=menu, grouped_classes=grouped_classes)


@app.route("/p_pretrainedClassification", methods=['POST', 'GET'])
def f_pretrainedClassification():
    if request.method == 'GET':
        return render_template('pretrainedClassification.html', title="Передовой ученый", menu=menu, class_model='')
    if request.method == 'POST':
        if(os.path.exists('static/media/pretrained_classification_blank.jpg')):
            os.remove('static/media/pretrained_classification_blank.jpg')

        img_file = flask.request.files.get('img', '')
        img_file.save('static/media/pretrained_classification_blank.jpg')

        img = image.load_img('static/media/pretrained_classification_blank.jpg', target_size=(96, 96))
        x = tf.keras.utils.img_to_array(img)
        x = tf.expand_dims(x, 0)
        pred = pretrained_model.predict_on_batch(x).flatten()
        pred = int(math.floor(float(pred[0] * len(classification_classes))))

        return render_template('pretrainedClassificationResult.html', title="Передовой ученый", menu=menu,
                               class_model=classification_classes_rus[classification_classes[pred]])


@app.route("/p_myClassification", methods=['POST', 'GET'])
def f_myClassification():
    if request.method == 'GET':
        return render_template('myClassification.html', title="Ученый-любитель", menu=menu, class_model='')
    if request.method == 'POST':
        if (os.path.exists('static/media/my_classification_blank.jpg')):
            os.remove('static/media/my_classification_blank.jpg')

        img_file = flask.request.files.get('img', '')
        img_file.save('static/media/my_classification_blank.jpg')

        img = image.load_img('static/media/my_classification_blank.jpg', target_size=(96, 96))
        x = tf.keras.utils.img_to_array(img)
        x = tf.expand_dims(x, 0)
        pred = own_model.predict(x).flatten()
        pred = tf.nn.softmax(pred)

        return render_template('myClassificationResult.html', title="Ученый-любитель", menu=menu,
                               class_model=classification_classes_rus[classification_classes[np.argmax(pred)]])


# API

@app.route('/api_yolov5sDetection', methods=['post'])
def api_yolov5sDetection():
    if (os.path.exists('static/media/detection_api_blank.jpg')):
        os.remove('static/media/detection_api_blank.jpg')

    img_file = flask.request.files.get('img', '')
    img_file.save('static/media/detection_api_blank.png')

    confidence = float(flask.request.form["confidence"])
    chosen_classes = flask.request.form.getlist("classes")

    chosen_classes_eng = []
    for classname in chosen_classes:
        chosen_classes_eng.append(detection_classes_eng[classname])

    img = image.load_img('static/media/detection_api_blank.png', target_size=(640, 640))

    generated = YOLOv5s_model(img)

    filtered_results = generated.pandas().xyxy[0]
    filtered_results = filtered_results[
        (filtered_results['name'].isin(chosen_classes_eng)) & (filtered_results['confidence'] >= confidence)]

    filtered_image = np.array(img)
    for _, row in filtered_results.iterrows():
        label = detection_classes_rus[row['name']]
        conf = row['confidence']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        color = detection_class_color[row['name']]
        cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_COMPLEX,
                    0.9, color, 3)

    filtered_image = Image.fromarray(filtered_image)

    img_io = io.BytesIO()
    filtered_image.save(img_io, 'JPEG')
    img_io.seek(0)
    print(img_io)

    return Response(img_io.read(), mimetype='image/jpeg', headers={'Content-Disposition': 'attachment; filename=detection_api_result.jpg'})


@app.route('/api_pretrainedClassification', methods=['post'])
def api_pretrainedClassification():
    if (os.path.exists('static/media/pretrained_classification_api_blank.jpg')):
        os.remove('static/media/pretrained_classification_api_blank.jpg')

    img_file = flask.request.files.get('img', '')
    img_file.save('static/media/pretrained_classification_api_blank.jpg')

    img = image.load_img('static/media/pretrained_classification_blank.jpg', target_size=(96, 96))
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)
    pred = pretrained_model.predict_on_batch(x).flatten()
    pred = int(math.floor(float(pred[0] * len(classification_classes))))

    return jsonify(item_class=classification_classes_rus[classification_classes[pred]])


@app.route('/api_myClassification', methods=['post'])
def api_myClassification():
    if (os.path.exists('static/media/my_classification_api_blank.jpg')):
        os.remove('static/media/my_classification_api_blank.jpg')

    img_file = flask.request.files.get('img', '')
    img_file.save('static/media/my_classification_api_blank.jpg')

    img = image.load_img('static/media/my_classification_blank.jpg', target_size=(96, 96))
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)
    pred = own_model.predict(x).flatten()
    pred = tf.nn.softmax(pred)

    return jsonify(item_class=classification_classes_rus[classification_classes[np.argmax(pred)]])


if __name__ == "__main__":
    app.run(debug=True)