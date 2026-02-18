import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = load_model("cifar10_model.keras")

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100

            prediction = f"{predicted_class} ({confidence:.2f}%)"

            return render_template("index.html",
                                   prediction=prediction,
                                   image_path=filepath)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
