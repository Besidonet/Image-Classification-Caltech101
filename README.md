# Image-Classification-Caltech101

This project is a simple Streamlit demo that classifies images using a fine-tuned ResNet-18 model trained on 10 classes from the Caltech101 dataset.

🔗 Live App

(https://image-classification-caltech101-umwmwjwyultckkjrtth7sd.streamlit.app/)

⚙️ Features:

Upload an image (JPG/PNG),
Automatic preprocessing (resize + normalize) and 
Prediction using a PyTorch ResNet-18 model

Shows:

Predicted class,
Confidence score and 
Top-3 class probabilities

🧠 Model Info

Backbone: ResNet-18 pretrained on ImageNet and 
Modified final FC layer → 10 output classes

Trained on these classes:

Faces, Faces_easy, Leopards, Motorbikes
airplanes, bonsai, car_side, chandelier
ketch, watch

📂 Project Files

app.py
requirements.txt
resnet18_caltech10.pth
class_names.json
README.md

▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

📦 Technologies

Python
PyTorch
Torchvision
Streamlit

📄 Notes

This project is for educational purposes only and demonstrates basic transfer learning + Streamlit deployment.
