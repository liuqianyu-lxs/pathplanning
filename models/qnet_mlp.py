import torch
import torch.nn as nn

class QNetMLP(nn.Module):
    def __init__(self):
        super(QNetMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 + 2 * 5 * 5, 256),  # 2 for goal_vector, 2 maps each 5x5 flattened
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state_image, goal_vector):
        # Flatten the state_image channels and concatenate with goal vector
        x = state_image.view(state_image.size(0), -1)  # [B, 2*5*5]
        x = torch.cat([x, goal_vector], dim=1)
        return self.fc(x)
