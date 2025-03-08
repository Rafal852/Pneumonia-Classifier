import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms, models

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for two classes
model.load_state_dict(torch.load('pneumonia_classifier.pth', map_location=torch.device('cpu')))
model.eval() 

@app.route('/')
def index():
    """Serve the HTML interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions."""
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        temp_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        image.save(temp_path)

        img = Image.open(temp_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        # Remove temp file after prediction
        os.remove(temp_path)

        # Return prediction result
        prediction = 'PNEUMONIA' if predicted.item() == 1 else 'NORMAL'
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
