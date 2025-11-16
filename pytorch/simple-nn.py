import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleNet().to(device)
# OR
# model = torch.nn.DataParallel(model)
# OR
# model = torch.nn.parallel.DistributedDataParallel(model)


# Feed-forward MLP
class SimpleNN(nn.Module):
    def __init__(self, features: int = 10, neurons: int = 50):
        super(SimpleNN, self).__init__()
        # Fully-connected layers
        self.fc1 = nn.Linear(features, neurons)
        self.fc2 = nn.Linear(neurons, 1)

    def forward(self, input: Any):
        tmp = torch.relu(self.fc1(input))
        return self.fc2(tmp)

model = SimpleNN(64, 64)
loss_fn = nn.MSELoss()
data = torch.randn(64, 10)
target = torch.randn(64, 10)

adamw = optim.AdamW(model.parameters, lr=1e-3, weight_decay=1e-2)
adamw_loss = []

for _ in range(20):
    adamw.zero_grad()
    out = model(data)
    loss = loss_fn(out, target)
    loss.backward()
    adamw.step()
    adamw_loss.append(loss.item())

