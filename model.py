import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNModel(nn.Module):
    def __init__(self) -> None:
        super(CNNModel, self).__init__()
        self.train_loss_list = []
        self.val_loss_list = []
        self.iteration_list = []
        self.accuracy_list = []
        # Conv1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)

        # Conv2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)

        # Max pool
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # activation function
        self.relu = nn.ReLU()

        # fully connected
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # conv1
        out = self.relu(self.cnn1(x))

        # max pool1
        out = self.maxpool(out)

        # conv2
        out = self.relu(self.cnn2(out))
        
        # max pool2
        out = self.maxpool(out)

        # fully connected
        # flatten
        if len(out.shape)>3:
            out = out.view(out.size(0), -1)
        else:
            out = torch.flatten(out)
        # print('out.shape = ', out.shape)
        return self.fc1(out)

    def train(self, train_loader, test_loader, learning_rate, batch_size, num_epochs):
        loss_f = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # TODO why 100? it should be batch_size? 
                train = Variable(images.view(batch_size, 1, 28, 28))
                labels = Variable(labels)
                optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
                optimizer.zero_grad()
                outputs = self(train)
                train_loss = loss_f(outputs, labels)
                train_loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    # Calculate Accuracy         
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        
                        test = Variable(images.view(batch_size, 1, 28, 28))
                        
                        # Forward propagation
                        outputs = self(test)
                        val_loss = loss_f(outputs, labels)
                        
                        # Get predictions from the maximum value
                        predicted = torch.max(outputs.data, 1)[1]
                        
                        # Total number of labels
                        total += len(labels)
                        
                        correct += (predicted == labels).sum()
                    
                    accuracy = 100 * correct / float(total)
                    
                    # store loss and iteration
                    self.val_loss_list.append(val_loss.data)
                    self.train_loss_list.append(train_loss.data)
                    self.iteration_list.append(i)
                    self.accuracy_list.append(accuracy)
                if i % 100 == 0:
                    # Print Loss
                    print('Epoch: {}  Iteration: {}  Train_Loss: {}  Val_Loss: {}  Accuracy: {} %'.format(epoch, i, train_loss.data, val_loss.data, accuracy))
