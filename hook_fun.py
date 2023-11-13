import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class MultiplyNetwork(nn.Module):
    def __init__(self):
        super(MultiplyNetwork, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input neuron, one output neuron

    def forward(self, x):
        return self.linear(x)

# Create an instance of the MultiplyNetwork
model = MultiplyNetwork()

# Define an input tensor and corresponding target (ground truth)
input_tensor = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor([[4.0], [9.0], [16.0]], dtype=torch.float32)  # Target is the element-wise multiplication of the input

# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Define an optimizer (e.g., Stochastic Gradient Descent - SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of training epochs
epochs = 100

# Training loop
for epoch in range(epochs):
    # Forward pass
    output = model(input_tensor)
    
    # Calculate the loss
    loss = criterion(output, target_tensor)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(input_tensor.grad.data)
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# After training, let's test the model
test_input = torch.tensor([[5.0]], dtype=torch.float32)
predicted_output = model(test_input)
print(f"Predicted Output for input {test_input.item()}: {predicted_output.item()}")
