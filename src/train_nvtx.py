import torch
import torchvision
import nvtx



# Load the Tiny ImageNet dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='/ibex/reference/CV/tinyimagenet/train',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
)

val_dataset = torchvision.datasets.ImageFolder(
    root='/ibex/reference/CV/tinyimagenet/val',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
)

test_dataset = torchvision.datasets.ImageFolder(
    root='/ibex/reference/CV/tinyimagenet/test',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
)

# use the correct device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device: ", device)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4
)

# Create a model
model = torchvision.models.resnet50(weights=None)
# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
# model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
torchvision.models

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# put model , train and test and val on the device
model.to(device)

for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):
        with nvtx.annotate("Batch" + str(i), color="green"):

            #load images and labels to device
            with nvtx.annotate("Copy to device", color="red"):
                images, labels = images.to(device), labels.to(device)

            # Forward pass
            with nvtx.annotate("Forward Pass", color="yellow"):
                outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagate the loss
            optimizer.zero_grad()

            with nvtx.annotate("Backward Pass", color="blue"):
                loss.backward()

            with nvtx.annotate("Optimizer step", color="orange"):
                optimizer.step()


            # Print the loss
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, batch {i + 1}/{len(train_loader)}, loss: {loss.item()}')
            # Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Val accuracy: {val_accuracy}')


# Test the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test accuracy: {test_accuracy}')