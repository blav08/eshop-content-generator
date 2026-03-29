import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trénuji na: {device}")

    # Data Augmentation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'dataset'
    if not os.path.exists(data_dir):
        print("Chyba: Složka dataset neexistuje!")
        return

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    classes = dataset.classes
    print(f"Kategorie ({len(classes)}): {classes}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)
    # Trénink
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 15

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}')

    # Uložení modelu
    torch.save(model.state_dict(), 'student_model.pth')
    print('Model uložen do student_model.pth')


if __name__ == '__main__':
    train()