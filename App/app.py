import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms, models
from flask import jsonify
import io

# ================= LOAD CSV FILES =================

disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# ================= MODEL =================

NUM_CLASSES = 39   # Change according to your dataset

model = models.resnet50(weights=None)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load trained weights
model.load_state_dict(
    torch.load("trained_model.pth", map_location=torch.device('cpu'))
)

model.eval()

# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= PREDICTION FUNCTION =================

def predict(image_path):

    img = Image.open(image_path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)

        predicted_class = torch.argmax(output, dim=1)

    return predicted_class.item()

# ================= FLASK =================

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# ================= ESP32cam =================
@app.route('/predict', methods=['POST'])
def predict_esp32():

    try:

        # Receive raw image bytes from ESP32-CAM
        image_bytes = request.data

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform image
        img = transform(image)

        img = img.unsqueeze(0)

        # Predict
        with torch.no_grad():

            output = model(img)

            pred = torch.argmax(output, dim=1).item()

        # Get disease + cure
        disease = disease_info['disease_name'][pred]

        cure = disease_info['Possible Steps'][pred]

        # Send JSON response to ESP32
        return jsonify({
            "disease": disease,
            "cure": cure
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })
# ================= SUBMIT =================

@app.route('/submit', methods=['GET', 'POST'])
def submit():

    if request.method == 'POST':

        image = request.files['image']

        filename = image.filename

        upload_folder = 'static/uploads'

        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)

        image.save(file_path)

        # Predict disease
        pred = predict(file_path)

        # Disease details
        title = disease_info['disease_name'][pred]

        description = disease_info['description'][pred]

        prevent = disease_info['Possible Steps'][pred]

        image_url = disease_info['image_url'][pred]

        # Supplement details
        supplement_name = supplement_info['supplement name'][pred]

        supplement_image_url = supplement_info['supplement image'][pred]

        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )

# ================= MARKET =================

@app.route('/market', methods=['GET', 'POST'])
def market():

    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# ================= MAIN =================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)