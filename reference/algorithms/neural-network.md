# Neural Networks

Used for classification and regression. They find complex patterns in unstructured data.

## Summary

- Supervised
- Input: a set of numerical features
- Output: probability value (for one or many classes)

## Concepts

- Jr: basic knowledge, training process, use cases
- Mid-Level: optimizers (Adam, RMSProp, SGD), exploding gradients, overfitting
- Sr: transformers, embeddings, backpropagation, regularization (AdamW, label smoothing)

## Use Cases

- Generative models
- Computer vision
- NLP (chatbots, speech recognition, language translation)
- Audio (speech-to-text, TTS, music gen)
- Self driving cars
- Search ranking

## Backpropgation

It's kind of like figuring out what discount you were given at the store, based on reading your receipt line by line.

Formally, it's reverse analysis of the computation graph generated during an iteration of training ("reverse mode automatic differentiation").

## Code Example

### PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self.__init__())
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = 10  # Number of features
hidden_size = 5  # Number of neurons in the hidden layer
output_size = 3  # Number of classes
learning_rate = 0.001
num_epochs = 20
batch_size = 32

X_train = ...
y_train = ...

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

## Vanishing and Exploding Gradients

Both are forms of extreme gradient behavior.

- Vanishing: the gradient becomes so small it effectively disappears
- Exploding: gradients grow too large, causing unstable updates, large weights, and inconsistent training results

Causes:

- Saturating activation functions, like sigmoid or tanh, squashing input values into too small a range (Vanishing)
- Gradient multiplication during backpropagation, many times over many layers, causing gradient shrinkage or exponential gradient growth (Vanishing/Exploding)

Fixes for vanishing gradients:

- Non-Saturating Activation Functions, like ReLU or leaky ReLU, which don't squash the input range
- Batch and Layer Normalization, which stabilize learning and help maintain gradients

Fixes for exploding gradients:

- Gradient Clipping: setting an explicit threshold
- The same fixes for vanishing gradients also apply

## Evaluation

IT REALLY DEPENDS.

## Limitations

- Complexity
- Low interpretability/explainability, lack of transparency
- Resource and time intensive
