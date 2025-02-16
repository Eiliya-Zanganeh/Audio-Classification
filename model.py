import torch.nn as nn
import torch.nn.functional as F


class AudioModel(nn.Module):
    def __init__(self, num_classes=27):
        super(AudioModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=40, stride=16, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=40, stride=1, padding=19)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=40, stride=1, padding=19)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=40, stride=1, padding=19)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=4)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=40, stride=1, padding=19)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(kernel_size=4)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (48000 // (4 ** 5 * 16)), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits
