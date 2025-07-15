from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Charger le modèle entraîné
model = load_model("model_voiture_professionnel.keras")

# Classes du modèle
class_names = ['back', 'front', 'left', 'right']   # ajuste si ton ordre est différent

# Taille d'image attendue
img_size = (224, 224)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image = None
    
    if request.method == "POST":
        # Récupérer le fichier
        file = request.files["file"]
        
        if file:
            uploaded_image = file.filename
            # Sauvegarder temporairement
            file_path = os.path.join("static", uploaded_image)
            file.save(file_path)
            
            # Préparer l'image
            img = image.load_img(file_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Prédiction
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            prediction = class_names[idx]
            confidence = float(preds[0][idx])
    
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_image=uploaded_image
    )

if __name__ == "__main__":
    app.run(debug=True)
