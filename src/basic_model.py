import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicSolarNet(nn.Module):
    """
    A simple CNN for 2-channel (AIA, HMI) solar image input.
    Input shape: (batch, 2, H, W)
    Output: regression or classification (customize as needed)
    Also stores observational properties for input (STEREO EUVI) and target (AIA).
    """
    def __init__(
        self,
        input_obs: dict,
        target_obs: dict,
        out_channels: int = 2
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_channels)

        # Observational properties for STEREO EUVI (input)
        self.input_obs = input_obs
        # Observational properties for AIA (target)
        self.target_obs = target_obs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = BasicSolarNet()
    dummy = torch.randn(4, 2, 128, 128)  # batch of 4, 2 channels, 128x128
    out = model(dummy)
    print("Output shape:", out.shape)
