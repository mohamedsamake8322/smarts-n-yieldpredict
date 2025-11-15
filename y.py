import torch
from torchvision import models, transforms
from PIL import Image

# ====== CONFIG ======
image_path = r"C:\Users\moham\Pictures\New folder\0ade14b6-8937-43ea-93eb-98343af6bae7___JR_HL 8026.JPG"
model_path = r"C:\smarts-n-yieldpredict.git\plant_disease_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes (dans le même ordre que ImageFolder)
classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 🔹 même taille que ton entraînement rapide
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====== CHARGEMENT IMAGE ======
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # ajoute batch dimension

# ====== CHARGEMENT MODELE ======
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ====== PREDICTION ======
with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    print(f"🟢 L'image est prédite comme : {classes[pred.item()]}")
