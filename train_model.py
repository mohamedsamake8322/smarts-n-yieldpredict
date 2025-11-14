import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def main():
    # ===============================
    # 1. CONFIG
    # ===============================
    data_dir = r"C:\Users\moham\Pictures\dataset_split"
    batch_size = 32
    num_epochs = 3  # 🔹 rapide pour test
    learning_rate = 0.001
    num_classes = 2  # Healthy / Bacterial Spot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # 2. TRANSFORMS (réduction images à 128x128)
    # ===============================
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),   # 🔹 taille réduite
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ===============================
    # 3. DATASETS & DATALOADERS
    # ===============================
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in ['train', 'val', 'test']
    }

    # ===============================
    # 4. MODELE (ResNet18)
    # ===============================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # ===============================
    # 5. PERTE & OPTIMISATEUR
    # ===============================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===============================
    # 6. ENTRAINEMENT
    # ===============================
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Sauvegarde du meilleur modèle
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "plant_disease_model.pth")

    print(f"✅ Entrainement terminé ! Meilleure précision en validation: {best_acc:.4f}")

    # ===============================
    # 7. TEST FINAL
    # ===============================
    model.load_state_dict(torch.load("plant_disease_model.pth"))
    model.eval()

    test_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    print(f"🎯 Précision sur le dataset test: {100 * test_corrects / total:.2f}%")

# Windows fix
if __name__ == "__main__":
    main()
