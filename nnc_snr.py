import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import networkx as nx
import numpy as np
from tqdm import tqdm

    
class SimpleFullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFullyConnected, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.batchnorm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.batchnorm(x)

        return x

class ButterflyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_std_dev):
        super(ButterflyNetwork, self).__init__()
        self.noise_std_dev = noise_std_dev
        
        self.A = SimpleFullyConnected(input_size, hidden_size, output_size)
        self.B = SimpleFullyConnected(input_size, hidden_size, output_size)
        self.C = SimpleFullyConnected(2 * output_size, hidden_size, output_size)
        self.D = SimpleFullyConnected(output_size, hidden_size, output_size)
        self.E = SimpleFullyConnected(2 * output_size, hidden_size, input_size)
        self.F = SimpleFullyConnected(2 * output_size, hidden_size, input_size)

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std_dev
        return x + noise

    def forward(self, x1, x2):

        x_A = self.A(x1)
        y_A = self.add_noise(x_A)
        x_B = self.B(x2)
        y_B = self.add_noise(x_B)
        
        x_C = self.C(torch.cat((y_A, y_B), dim=-1))
        y_C = self.add_noise(x_C)
        
        x_D = self.D(y_C)
        y_D = self.add_noise(x_D)
        
        x_E = self.E(torch.cat((y_A, y_D), dim=-1))
        x_F = self.F(torch.cat((y_B, y_D), dim=-1))
        
        y1 = x_E
        y2 = x_F
        
        return y1, y2, [x_A, x_B, x_C, x_D, x_E, x_F]

# general graph?
# class GraphButterflyNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, noise_std_dev, adjacency_matrix):
#         super(GraphButterflyNetwork, self).__init__()
#         self.noise_std_dev = noise_std_dev
#         self.adjacency_matrix = adjacency_matrix
#         self.num_nodes = len(adjacency_matrix)

#         # Create a ModuleList of FullyConnected layers for each node
#         self.nodes = nn.ModuleList([
#             FullyConnected(input_size, hidden_size, output_size)
#             for _ in range(self.num_nodes)
#         ])

#         # Compute the topological order of the nodes
#         graph = nx.DiGraph(np.array(adjacency_matrix).T)
#         self.topological_order = list(nx.topological_sort(graph))

#     def add_noise(self, x):
#         # Add Gaussian noise to the input tensor
#         noise = torch.randn_like(x) * self.noise_std_dev
#         return x + noise

#     def forward(self, x1, x2):
#         outputs = [None] * self.num_nodes
#         outputs[0] = self.nodes[0](x1)
#         outputs[1] = self.nodes[1](x2)

#         for i in self.topological_order[2:]:
#             # Find the indices of the parent nodes based on the adjacency matrix
#             parent_indices = [j for j, value in enumerate(self.adjacency_matrix[i]) if value == 1]
            
#             # Concatenate the outputs of the parent nodes
#             input_tensor = torch.cat([outputs[j] for j in parent_indices], dim=-1)

#             # Pass the input through the current node without noise
#             outputs[i] = self.nodes[i](input_tensor)

#             # Add noise to the output
#             outputs[i] = self.add_noise(outputs[i])

#         y1, y2 = outputs[-2], outputs[-1]
#         return y1, y2, outputs[:-2]

# def custom_loss(output, target, adjacency_matrix, y_list):
#     # Calculate the binary cross-entropy loss
#     # bce_loss = nn.BCEWithLogitsLoss()(output, target)
#     mse_loss = nn.MSELoss()(output, target)
#     expectation_loss = 0.0

#     # Calculate the regularization term based on the adjacency matrix and lambda matrix
#     for i in range(len(adjacency_matrix)):
#         for j in range(len(adjacency_matrix[i])):
#             if adjacency_matrix[i][j] == 1:
#                 expectation_loss += lambda_matrix[i][j] * torch.mean(y_list[i] ** 2)

#     # Return the total loss as the sum of the binary cross-entropy loss and the regularization term
#     return mse_loss + expectation_loss

def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        
        x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
        x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
        
        y1, y2, y_list = model(x1, x2)
        
        y = torch.cat((y1, y2), dim=1)
        target = torch.cat((x1, x2), dim=1)
        
        # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
        # loss = nn.BCEWithLogitsLoss()(y, target)
        loss = nn.MSELoss()(y, target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def test(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
            x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
            
            y1, y2, y_list = model(x1, x2)
            
            y = torch.cat((y1, y2), dim=1)
            target = torch.cat((x1, x2), dim=1)
            
            # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
            # loss = nn.BCEWithLogitsLoss()(y, target)
            loss = nn.MSELoss()(y, target)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

if __name__ == '__main__':

    training = True
    device = torch.device("mps")
    epochs = 150
    batch_size = 32
    input_size = 14 * 28
    hidden_size = 256
    output_size = 256
    noise_std_dev = np.sqrt(0.0001)

    power = 7
    eq_noise_std_dev = noise_std_dev / np.sqrt(power)
    learning_rate = 0.001
    
    adjacency_matrix = torch.tensor([
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ], dtype=torch.float32)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ButterflyNetwork(input_size, hidden_size, output_size, eq_noise_std_dev).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if training:
        for epoch in tqdm(range(epochs)):
            train_loss = train(model, train_loader, optimizer, device)
            test_loss = test(model, test_loader, device)
            
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            # save best model
            if epoch == 0:
                best_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), 'best_model.pt')

        print(f"Best Test Loss: {best_loss:.4f}")
        print("Model saved at best_model.pt")

    # testing
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    test_loss = test(model, test_loader, device)

    # get maximum pixel value in test set
    

    
