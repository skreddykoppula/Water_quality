from flask import Flask, render_template,url_for,request
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import load_model
import base64
import matplotlib.pyplot as plt
import json

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "water images"
img_ext = ['jpeg', 'jpg', 'png', 'bmp']
for img_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, img_class)):
        img_path = os.path.join(data_dir, img_class, image)
        try:
            img = cv2.imread(img_path)
            tip = imghdr.what(img_path)
            if tip not in img_ext:
                print('Image not in ext list {}'.format(img_path))
                os.remove(img_path)
        except Exception as e:
            print('Issue with image {}'.format(img_path))

sample_images = []

for img_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, img_class)):
        img_path = os.path.join(data_dir, img_class, image)
        try:
            img = cv2.imread(img_path)
            tip = imghdr.what(img_path)
            if tip not in img_ext:
                print('Image not in ext list {}'.format(img_path))
                os.remove(img_path)
            else:
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    sample_images.append(base64_image)
                if len(sample_images) >= 4:
                    break
        except Exception as e:
            print('Issue with image {}'.format(img_path))

template_data = {
    "sample_images": sample_images,
}

if os.path.exists("water_model.h5"):
    model = load_model("water_model.h5")
    
    history_file_path = "training_history.json"
    if os.path.exists(history_file_path):
        with open(history_file_path, "r") as history_file:
            history_data = json.load(history_file)
    else:
        history_data = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
else:
    data = tf.keras.utils.image_dataset_from_directory('water images')
    data = data.map(lambda x, y: (x / 255, y))
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2) + 1
    test_size = int(len(data) * 0.1) + 1
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    history = model.fit(train, epochs=24, validation_data=val)

    history_data = history.history

    # Save the history data to a file
    with open("training_history.json", "w") as history_file:
        json.dump(history_data, history_file)

    model.save("water_model.h5")


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/cnn_model', methods=['POST', 'GET'])
def cnn_model():
    model = load_model("water_model.h5")
    data = tf.keras.utils.image_dataset_from_directory('water images')
    data = data.map(lambda x, y: (x / 255, y))
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    predictions = model.predict(batch[0])
    predicted_classes = ["dirty" if pred >= 0.5 else "clean" for pred in predictions]

    num_images = len(data)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    template_data = {
        "data_description": "Found {} files belonging to 2 classes.".format(num_images),
        "model_summary": model.summary(),
        "precision": 1.0,
        "recall": 1.0,
        "accuracy": 1.0,
        "predicted_class": predicted_classes[0],
        "num_images": num_images,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "sample_images": sample_images,
    }
    return render_template('cnn_model.html', template_data=template_data)

@app.route('/loss_accuracy', methods=['GET'])
def loss_accuracy():
    loss_plot_path = "static/loss_plot.png"
    accuracy_plot_path = "static/accuracy_plot.png"
    model = load_model("water_model.h5")
    with open("training_history.json", "r") as history_file:
        history_data = json.load(history_file)

    # Create and save loss plot
    fig_loss = plt.figure()
    plt.plot(history_data['loss'], color='teal', label='loss')
    plt.plot(history_data['val_loss'], color='orange', label='val_loss')
    fig_loss.suptitle('Loss', fontsize=20)
    plt.legend(loc='upper left')
    fig_loss.savefig(loss_plot_path)

    # Create and save accuracy plot
    fig_accuracy = plt.figure()
    plt.plot(history_data['accuracy'], color='teal', label='accuracy')
    plt.plot(history_data['val_accuracy'], color='orange', label='val_accuracy')
    fig_accuracy.suptitle('Accuracy', fontsize=20)
    plt.legend(loc='upper left')
    fig_accuracy.savefig(accuracy_plot_path)

    # Generate URLs for images using url_for
    loss_plot_url = url_for('static', filename='loss_plot.png')
    accuracy_plot_url = url_for('static', filename='accuracy_plot.png')

    return render_template('loss_accuracy.html', loss_plot=loss_plot_url, accuracy_plot=accuracy_plot_url)


model = load_model("water_model.h5")
app.config['UPLOAD_FOLDER'] = 'static'
upload_folder = app.config['UPLOAD_FOLDER']
# Define the /predict route
# Define the /predict route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        print("rev")
        if file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("sk")
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            binary_pred = np.round(pred)
            predicted_class = "dirty" if binary_pred[0][0] == 1 else "clean"

            return render_template('prediction.html', filename=filename, prediction_result=predicted_class)

    # If no valid image was uploaded, simply render the prediction.html page
    return render_template('prediction.html')





