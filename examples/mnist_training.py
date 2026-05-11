import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import intel_npu_acceleration as npu
import time
import os

# --- Model Definition ---

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # We still use standard layers for parameter storage
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # We use explicit NPU ops for computation
        # Since functional.py is autograd-aware, this works for training!
        x = npu.conv2d(x, self.conv1.weight, self.conv1.bias, 
                       stride=self.conv1.stride, padding=self.conv1.padding)
        x = npu.relu(x)
        
        x = npu.conv2d(x, self.conv2.weight, self.conv2.bias,
                       stride=self.conv2.stride, padding=self.conv2.padding)
        x = npu.relu(x)
        
        x = npu.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        
        x = npu.linear(x, self.fc1.weight, self.fc1.bias)
        x = npu.relu(x)
        
        x = npu.linear(x, self.fc2.weight, self.fc2.bias)
        output = npu.softmax(x, dim=1)
        return torch.log(output + 1e-10) # LogSoftmax poorman's impl

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    print(f"\nTraining on Hybrid NPU (Forward) + CPU (Backward)...")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # This will now use NPU accelerated ops if wrapped properly, 
        # but since we want to show 'manual' use of NPU ops in training:
        output = model(data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test_npu(compiled_model, test_loader):
    print("\nEvaluating on Intel NPU...")
    start_time = time.time()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Note: compiled_model expects inputs on CPU, it handles the NPU transfer
            output = compiled_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    elapsed = time.time() - start_time
    print(f'NPU Test set: Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')
    print(f'Total NPU Inference Time: {elapsed:.3f}s')
    return elapsed

def test_cpu(model, test_loader):
    print("\nEvaluating on CPU...")
    model.eval()
    start_time = time.time()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    elapsed = time.time() - start_time
    print(f'CPU Test set: Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')
    print(f'Total CPU Inference Time: {elapsed:.3f}s')
    return elapsed

def main():
    # Training on CPU
    device = torch.device("cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    print("Loading MNIST dataset...")
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    print("\nStep 1: Training model on CPU for 1 epoch...")
    # Training on NPU (Backprop) is currently on the roadmap.
    # We use CPU for training and NPU for accelerated inference.
    train(model, device, train_loader, optimizer, 1)

    print("\nStep 2: Compiling model for Intel NPU...")
    model.eval()
    example_input = torch.randn(1, 1, 28, 28)
    
    # Check if NPU is available before compiling
    if not npu.is_available():
        print("⚠️ Warning: Intel NPU not detected. Compiler will fallback to CPU (OpenVINO optimized).")
    
    npu_model = npu.compile(model, example_input)

    # Step 3: Compare Performance
    cpu_time = test_cpu(model, test_loader)
    npu_time = test_npu(npu_model, test_loader)

    speedup = cpu_time / npu_time if npu_time > 0 else 0
    print(f"\n🚀 NPU Speedup: {speedup:.2f}x")

if __name__ == '__main__':
    main()
