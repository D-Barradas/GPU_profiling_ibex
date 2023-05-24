import torch 
#load activation function
import torch.nn.functional as F
#load optimizer
import torch.optim as optim
# load the CIFAR10 dataset
from torchvision import datasets, transforms , models 
# load DataLolader
from torch.utils.data import DataLoader
import argparse

transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,)),
])

# define a funtion to count the number of gpus available
def get_num_gpus():
    return torch.cuda.device_count()

# def get_args():
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
parser.add_argument('--num_workers', default=5, type=int, metavar='N',
                        help='number of workers (default: 5)')
parser.add_argument('--log-dir', default='log', type=str, metavar='PATH',
                        help='path to log')
parser.add_argument('--warmup-epochs', default=0.0, type=float, metavar='N',
                        help='number of warmup epochs (default: 0.0)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                        help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='seed for initializing training. ')
parser.add_argument('--print_freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    

args = parser.parse_args()
# # if args are incomplete print the usage 
if args is None:
    print("Usage: python train.py --epochs 5 --batch_size 256 --num_workers 5 --log-dir log --warmup-epochs 0.0 --lr 0.1 --momentum 0.9 --wd 1e-4 --seed None --print_freq 10")
    exit(1)

# print all the args recieved
# print("args: ", args)

def load_data():
    # load data using DataLoader and have a batch size of args.batch_size from the directory "/ibex/reference/CV/CIFAR/cifar-10-batches-py"
    trainset = datasets.CIFAR10(root="/ibex/reference/CV/CIFAR/", download=False, train=True, transform=transform )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testset = datasets.CIFAR10(root="/ibex/reference/CV/CIFAR/", download=False, train=False, transform=transform )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    return trainloader, testloader

def train_model(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    steps = 0
    # change to cuda
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every))
                running_loss = 0

# test the model
def test_model(model, testloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("Test loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test accuracy: {:.3f}".format(accuracy/len(testloader)))

# define the model fror CIFAR10 using resnet50
# The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
resnet50 = models.resnet50(weights=None, progress=True)



# execute the code
trainloader, testloader = load_data()
model = resnet50
criterion = torch.nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)
# scale learning rate by the number of GPUs.
lr_scaler = 1.0
if get_num_gpus() > 1:
    lr_scaler = 1.0 / get_num_gpus()
else:
    lr_scaler = 1.0

optimizer = optim.SGD(model.parameters(),
                      lr=(args.lr *lr_scaler),
                      momentum=args.momentum, weight_decay=args.wd)

train_model(model, trainloader, epochs=args.epochs , print_every=args.print_freq , criterion=criterion, optimizer=optimizer , device='cuda')
test_model(model, testloader, criterion, 'cuda')