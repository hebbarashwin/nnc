import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import networkx as nx
import argparse
import random
import numpy as np
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser(description='Butterfly Network Training')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument(    '--input_size', type=int, default=14 * 28, help='Input size for the model')
    parser.add_argument(    '--hidden_size', type=int, default=256, help='Hidden size for the model')
    parser.add_argument(    '--output_size', type=int, default=256, help='Output size for the model')
    parser.add_argument('-n', '--noise_sigma', type=float, default=0.01, help='Noise standard deviation')
    parser.add_argument('-p', '--power', type=int, default=7, help='Power for eq_noise_std_dev calculation')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to use for training (cpu, cuda, mps)')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to save and load the best model')
    parser.add_argument('--test', action='store_true', help='Only test the model')

    args = parser.parse_args()

    if args.model_path is None:
        rand_num = random.randint(10000, 99999)
        args.model_path = f'{rand_num}'

    return args

    
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

    progress = tqdm(dataloader, leave=False, desc="Batches")
    for images, labels in progress:
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
    running_loss /= len(dataloader)
    psnr = 10*np.log10(torch.max(images)**2 / running_loss)
    
    return running_loss, psnr

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
    running_loss = running_loss / len(dataloader)
    psnr = 10*np.log10(torch.max(images)**2 / running_loss)
    
    return running_loss, psnr

def main(args):

    model_path = os.path.join('models', args.model_path, 'model.pt')
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Model path : {model_path}")



    # change device to cuda or cpu .. mps is for mac-m1 gpu
    if args.device == 'cpu':
        device = torch.device("cpu") 
    elif args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'mps':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
       
    epochs = 150
    eq_noise_sigma = args.noise_sigma / np.sqrt(args.power)    
    # adjacency_matrix = torch.tensor([
    #     [0, 0, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 1]
    # ], dtype=torch.float32)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ButterflyNetwork(args.input_size, args.hidden_size, args.output_size, eq_noise_sigma).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if not args.test:
        with tqdm(range(args.epochs), desc="Epochs") as epoch_progress:
            for epoch in epoch_progress:
                train_loss, train_psnr = train(model, train_loader, optimizer, device)
                test_loss, test_psnr = test(model, test_loader, device)

                epoch_progress.set_postfix({"Test Loss": test_loss, "Test PSNR": test_psnr})
                epoch_progress.write(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}')

                # save best model
                if epoch == 0:
                    best_loss = test_loss
                    torch.save(model.state_dict(), model_path)
                else:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        torch.save(model.state_dict(), model_path)

        print(f"Best Test Loss: {best_loss:.4f}")
        print("Model saved at best_model.pt")

    # testing
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_loss, test_psnr = test(model, test_loader, device)
        print(f"Loaded model, Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}")
    else:
        print(f"No model found at {model_path}")


    
if __name__ == '__main__':
    args = get_args()
    main(args)
