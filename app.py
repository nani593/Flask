from flask import Flask, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder="templates")
model = load_model('disaster.h5')

cap = None  # Initialize cap here

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/intro', methods=['GET'])
def intro():
    return render_template('intro.html')

@app.route('/upload', methods=['GET', 'POST'])
def predict():
    global cap  # Use the global cap

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        x = np.expand_dims(frame, axis=0)
        result = np.argmax(model.predict(x), axis=-1)
        index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
        result = str(index[result[0]])
        cv2.putText(output, "activity:{}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.imshow("Output", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        cap = None  # Reset cap

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
