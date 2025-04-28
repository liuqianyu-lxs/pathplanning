import torch
import torch.nn as nn

class QNetCNN(nn.Module):
    def __init__(self):
        super(QNetCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state_image, goal_vector):
        x = self.conv(state_image)
        x = torch.cat([x, goal_vector], dim=1)
        return self.fc(x)
