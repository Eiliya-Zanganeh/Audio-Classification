import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import sum, argmax
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import AudioModel
from dataset import generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} | Device Name: {device_name}')

train_dataset, validation_dataset, test_dataset = generate_dataset()

model = AudioModel().to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = CrossEntropyLoss()


def train():
    for epoch in range(50):
        model.train()
        train_loss = 0
        train_accuracy = 0
        total_batch = 0

        for batch in train_dataset:
            optimizer.zero_grad()
            audios, labels = batch
            audios, labels = audios.to(device), labels.to(device)
            outputs = model(audios)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            outputs = argmax(outputs, dim=1)
            accuracy = sum(outputs == labels).item() / len(labels)

            train_loss += loss.item()
            train_accuracy += accuracy
            total_batch += 1

        print(f'Epoch [{epoch + 1}/50]')
        print(f'Train Loss: {train_loss / total_batch: .4f} | Train Accuracy: {train_accuracy / total_batch: .4f}')

        validation_loss = 0
        validation_accuracy = 0
        total_batch = 0

        model.eval()
        for batch in validation_dataset:
            audios, labels = batch
            audios, labels = audios.to(device), labels.to(device)
            outputs = model(audios)
            loss = criterion(outputs, labels)

            outputs = argmax(outputs, dim=1)
            accuracy = sum(outputs == labels).item() / len(labels)

            validation_loss += loss.item()
            validation_accuracy += accuracy
            total_batch += 1

        print(
            f'Validation Loss: {validation_loss / total_batch:.4f} | Validation Accuracy: {validation_accuracy / total_batch:.4f}')
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved')


def test():
    model.eval()
    test_loss = 0
    test_accuracy = 0
    total_batches = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_dataset:
            audios, labels = batch
            audios, labels = audios.to(device), labels.to(device)

            outputs = model(audios)
            loss = criterion(outputs, labels)

            preds = argmax(outputs.data, 1)

            accuracy = 100 * sum(labels == preds).item() / len(labels)

            test_accuracy += accuracy
            test_loss += loss.item()
            total_batches += 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_accuracy /= total_batches
    test_loss /= total_batches

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [f'Class {i}' for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')