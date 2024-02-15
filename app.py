from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction_result="No se seleccionó ninguna imagen.")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', prediction_result="No se seleccionó ninguna imagen.")

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Cargar el modelo RNN previamente entrenado
        rnn = load_model('dataset_cnn_model.h5')

        # Cargar y preprocesar la imagen para realizar la clasificación
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizar los valores de píxeles al rango [0, 1]

        # Realizar la clasificación
        prediction = rnn.predict(img_array)

        # Interpretar el resultado de la clasificación
        # Interpretar el resultado de la clasificación
        class_labels = ['Carro', 'Moto', 'Avion', 'Tren']
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Utilizar la etiqueta predicha para mostrar el resultado
        if 0 <= predicted_class_index < len(class_labels):
            prediction_result = f"La imagen corresponde a un: {class_labels[predicted_class_index]}"
        else:
            prediction_result = "No se pudo determinar la clase correctamente."


        return render_template("index.html", prediction_result=prediction_result)

    else:
        return render_template('index.html', prediction_result="Formato de archivo no permitido.")

if __name__ == "__main__":
    app.run()
