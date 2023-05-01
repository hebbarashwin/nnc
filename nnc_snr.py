import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import networkx as nx
import argparse
import random
import numpy as np
from tqdm import tqdm
import os

def snr_sigma2db(sigma):
    return 10 * np.log10(1 / (sigma ** 2))

def snr_db2sigma(snr):
    return 10 ** (-snr *1.0/ 20)


def get_args():
    parser = argparse.ArgumentParser(description='Butterfly Network Training')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument(    '--input_size', type=int, default=14 * 28, help='Input size for the model')
    parser.add_argument(    '--hidden_size', type=int, default=256, help='Hidden size for the model')
    parser.add_argument(    '--output_size', type=int, default=256, help='Output size for the model')
    parser.add_argument('-t_s', '--train_snr', type=float, default=40, help='Training SNR (40dB : sigma = 0.01)')
    parser.add_argument('-v_s', '--val_snr', type=float, default=40, help='Validation SNR (40dB : sigma = 0.01)')
    parser.add_argument('-p', '--power', type=int, default=1, help='Power for eq_noise_std_dev calculation')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to use for training (cpu, cuda, mps)')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to save and load the best model')
    parser.add_argument('--dropout', action='store_true', help='Add dropout during training')
    parser.add_argument('--test', action='store_true', help='Only test the model')
    parser.add_argument('--scale_outputs', action='store_true', help='Scale outputs like paper')
    args = parser.parse_args()

    if args.model_path is None:
        rand_num = random.randint(10000, 99999)
        args.model_path = f'{rand_num}'

    return args

def link_dropout(x, p):
    # x - batched input (first dimension is batch)
    assert 0 <= p <= 1

    # Generate a random number for each example in the batch
    random_numbers = torch.rand(x.size(0), 1, device=x.device)
    # Create a mask with the same size as the input_tensor
    dropout_mask = (random_numbers >= p).float()
    # Expand the dropout mask to match the shape of the input_tensor
    dropout_mask = dropout_mask.expand_as(x)
    # Apply the dropout mask to the input_tensor
    output_tensor = x * dropout_mask
    
    return output_tensor
    
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, num_bits):
        min_val = tensor.min()
        max_val = tensor.max()
        qmin = 0
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        q_tensor = torch.clamp((tensor - min_val) / scale + qmin, qmin, qmax)
        q_tensor = torch.round(q_tensor) * scale + min_val
        ctx.save_for_backward(tensor, q_tensor)
        return q_tensor

    @staticmethod
    def backward(ctx, grad_output):
        tensor, q_tensor = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None

def quantize_output(tensor, num_bits):
    return QuantizeSTE.apply(tensor, num_bits)

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
    def __init__(self, input_size, hidden_size, output_size, noise_std_dev, scale_power = 1):
        super(ButterflyNetwork, self).__init__()
        self.noise_std_dev = noise_std_dev
        self.scale_power = scale_power

        self.A = SimpleFullyConnected(input_size, hidden_size, output_size)
        self.B = SimpleFullyConnected(input_size, hidden_size, output_size)
        self.C = SimpleFullyConnected(2 * output_size, hidden_size, output_size)
        self.D = SimpleFullyConnected(output_size, hidden_size, output_size)
        self.E = SimpleFullyConnected(2 * output_size, hidden_size, 2*input_size)
        self.F = SimpleFullyConnected(2 * output_size, hidden_size, 2*input_size)

        # self.dropout = nn.Dropout(0.1)
        self.dropout = lambda x: link_dropout(x, 0.1)

    def add_noise(self, x, noise_std_dev):

        noise = torch.randn_like(x) * noise_std_dev
        return x + noise

    def forward(self, x1, x2, noise_std_dev = None, dropped = None):

        if dropped is None:
            dropped = []

        if noise_std_dev is None:
            noise_std_dev = float(self.noise_std_dev)  # Ensure that noise_std_dev is a scalar (float)


        x_A = self.A(x1) * self.scale_power
        y_AC = self.add_noise(self.dropout(x_A) if 'AC' in dropped else x_A, noise_std_dev)
        y_AE = self.add_noise(self.dropout(x_A) if 'AE' in dropped else x_A, noise_std_dev)

        x_B = self.B(x2) * self.scale_power
        y_BC = self.add_noise(self.dropout(x_B) if 'BC' in dropped else x_B, noise_std_dev)
        y_BF = self.add_noise(self.dropout(x_B) if 'BF' in dropped else x_B, noise_std_dev)

        x_C = self.C(torch.cat((y_AC, y_BC), dim=-1)) * self.scale_power
        y_CD = self.add_noise(self.dropout(x_C) if 'CD' in dropped else x_C, noise_std_dev)

        x_D = self.D(y_CD) * self.scale_power
        y_DE = self.add_noise(self.dropout(x_D) if 'DE' in dropped else x_D, noise_std_dev)
        y_DF = self.add_noise(self.dropout(x_D) if 'DF' in dropped else x_D, noise_std_dev)

        x_E = self.E(torch.cat((y_AE, y_DE), dim=-1)) * self.scale_power
        x_F = self.F(torch.cat((y_BF, y_DF), dim=-1)) * self.scale_power

        y1 = x_E
        y2 = x_F

        return y1, y2, [x_A, x_B, x_C, x_D, x_E, x_F]
    
    def run_drop_link(self, x1, x2, dropped = [], noise_std_dev = None):

        if noise_std_dev is None:
            noise_std_dev = self.noise_std_dev
        x_A = self.A(x1) * self.scale_power

        if 'AC' not in dropped:
            y_AC = self.add_noise(x_A, noise_std_dev)
        else:
            y_AC = self.add_noise(torch.zeros_like(x_A, device=x_A.device), noise_std_dev)

        if 'AE' not in dropped:
            y_AE = self.add_noise(x_A, noise_std_dev)
        else:
            y_AE = self.add_noise(torch.zeros_like(x_A, device=x_A.device), noise_std_dev)

        x_B = self.B(x2) * self.scale_power

        if 'BC' not in dropped:
            y_BC = self.add_noise(x_B, noise_std_dev)
        else:
            y_BC = self.add_noise(torch.zeros_like(x_B, device=x_B.device), noise_std_dev)

        if 'BF' not in dropped:
            y_BF = self.add_noise(x_B, noise_std_dev)
        else:
            y_BF = self.add_noise(torch.zeros_like(x_B, device=x_B.device), noise_std_dev)

        x_C = self.C(torch.cat((y_AC, y_BC), dim=-1)) * self.scale_power

        if 'CD' not in dropped:
            y_CD = self.add_noise(x_C, noise_std_dev)
        else:
            y_CD = self.add_noise(torch.zeros_like(x_C, device=x_C.device), noise_std_dev)

        x_D = self.D(y_CD) * self.scale_power

        if 'DE' not in dropped:
            y_DE = self.add_noise(x_D, noise_std_dev)
        else:
            y_DE = self.add_noise(torch.zeros_like(x_D, device=x_D.device), noise_std_dev)

        if 'DF' not in dropped:
            y_DF = self.add_noise(x_D, noise_std_dev)
        else:
            y_DF = self.add_noise(torch.zeros_like(x_D, device=x_D.device), noise_std_dev)

        x_E = self.E(torch.cat((y_AE, y_DE), dim=-1)) * self.scale_power
        x_F = self.F(torch.cat((y_BF, y_DF), dim=-1)) * self.scale_power

        y1 = x_E
        y2 = x_F

        return y1, y2, [x_A, x_B, x_C, x_D, x_E, x_F]

    def run_quantized(self, num_bits, x1, x2, noise_std_dev = None, dropped = None):

        if dropped is None:
            dropped = []

        if noise_std_dev is None:
            noise_std_dev = float(self.noise_std_dev)  # Ensure that noise_std_dev is a scalar (float)


        x_A = self.A(x1) * self.scale_power
        x_A = quantize_output(x_A, num_bits)
        y_AC = self.add_noise(self.dropout(x_A) if 'AC' in dropped else x_A, noise_std_dev)
        y_AE = self.add_noise(self.dropout(x_A) if 'AE' in dropped else x_A, noise_std_dev)

        x_B = self.B(x2) * self.scale_power
        x_B = quantize_output(x_B, num_bits)
        y_BC = self.add_noise(self.dropout(x_B) if 'BC' in dropped else x_B, noise_std_dev)
        y_BF = self.add_noise(self.dropout(x_B) if 'BF' in dropped else x_B, noise_std_dev)

        x_C = self.C(torch.cat((y_AC, y_BC), dim=-1)) * self.scale_power
        x_C = quantize_output(x_C, num_bits)
        y_CD = self.add_noise(self.dropout(x_C) if 'CD' in dropped else x_C, noise_std_dev)

        x_D = self.D(y_CD) * self.scale_power
        x_D = quantize_output(x_D, num_bits)
        y_DE = self.add_noise(self.dropout(x_D) if 'DE' in dropped else x_D, noise_std_dev)
        y_DF = self.add_noise(self.dropout(x_D) if 'DF' in dropped else x_D, noise_std_dev)

        x_E = self.E(torch.cat((y_AE, y_DE), dim=-1)) * self.scale_power
        x_F = self.F(torch.cat((y_BF, y_DF), dim=-1)) * self.scale_power

        y1 = x_E
        y2 = x_F

        return y1, y2, [x_A, x_B, x_C, x_D, x_E, x_F]

# general graph?
class GraphButterflyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_std_dev, adjacency_matrix):
        super(GraphButterflyNetwork, self).__init__()
        self.noise_std_dev = noise_std_dev
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = len(adjacency_matrix)

        # Create a ModuleList of FullyConnected layers for each node
        self.nodes = nn.ModuleList([
            SimpleFullyConnected(input_size, hidden_size, output_size)
            for _ in range(self.num_nodes)
        ])

        # Compute the topological order of the nodes
        graph = nx.DiGraph(np.array(adjacency_matrix).T)
        self.topological_order = list(nx.topological_sort(graph))

    def add_noise(self, x):
        # Add Gaussian noise to the input tensor
        noise = torch.randn_like(x) * self.noise_std_dev
        return x + noise

    def forward(self, x1, x2):
        outputs = [None] * self.num_nodes
        outputs[0] = self.nodes[0](x1)
        outputs[1] = self.nodes[1](x2)

        for i in self.topological_order[2:]:
            # Find the indices of the parent nodes based on the adjacency matrix
            parent_indices = [j for j, value in enumerate(self.adjacency_matrix[i]) if value == 1]
            
            # Concatenate the outputs of the parent nodes
            input_tensor = torch.cat([outputs[j] for j in parent_indices], dim=-1)

            # Pass the input through the current node without noise
            outputs[i] = self.nodes[i](input_tensor)

            # Add noise to the output
            outputs[i] = self.add_noise(outputs[i])

        y1, y2 = outputs[-2], outputs[-1]
        return y1, y2, outputs[:-2]

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

def train(model, dataloader, optimizer, device, dropped_list = None):
    model.train()
    running_loss = 0.0

    progress = tqdm(dataloader, leave=False, desc="Batches")
    for images, labels in progress:
        optimizer.zero_grad()
        
        x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
        x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
        
        y1, y2, y_list = model(x1, x2, dropped = dropped_list)        
        # y = torch.cat((y1, y2), dim=1)
        target = torch.cat((x1, x2), dim=1)
        
        # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
        # loss = nn.BCEWithLogitsLoss()(y, target)
        loss = nn.MSELoss()(y1, target) + nn.MSELoss()(y2, target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    running_loss /= len(dataloader)
    psnr = 10*np.log10(torch.max(images)**2 / running_loss)
    
    return running_loss, psnr

def test(model, dataloader, test_sigma_range, device, quantize = False, num_bits = 8):
    
    results = []
    model.eval()    
    with torch.no_grad():
        for sigma in test_sigma_range:
            snr = snr_sigma2db(sigma)
            running_loss = 0.0
            for images, labels in dataloader:
                x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                
                if not quantize:
                    y1, y2, y_list = model(x1, x2, noise_std_dev=sigma)
                else:
                    y1, y2, y_list = model.run_quantized(num_bits, x1, x2, noise_std_dev=sigma)
                
                y = torch.cat((y1, y2), dim=1)
                target = torch.cat((x1, x2), dim=1)
                
                # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # loss = nn.BCEWithLogitsLoss()(y, target)
                loss = nn.MSELoss()(y, target)
                
                running_loss += loss.item()
            running_loss = running_loss / len(dataloader)
            psnr = 10*np.log10(torch.max(images)**2 / running_loss).item()
            results.append((snr, running_loss, psnr))

    return results

def test_dropped(model, dataloader, test_sigma_range, device, dropped = []):
    
    results = []
    model.eval()    
    with torch.no_grad():
        for sigma in test_sigma_range:
            snr = snr_sigma2db(sigma)
            running_loss = 0.0
            for images, labels in dataloader:
                x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                
                y1, y2, y_list = model.run_drop_link(x1, x2, noise_std_dev=sigma, dropped=dropped)
                
                y = torch.cat((y1, y2), dim=1)
                target = torch.cat((x1, x2), dim=1)
                
                # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # loss = nn.BCEWithLogitsLoss()(y, target)
                loss = nn.MSELoss()(y, target)
                
                running_loss += loss.item()
            running_loss = running_loss / len(dataloader)
            psnr = 10*np.log10(torch.max(images)**2 / running_loss).item()
            results.append((snr, running_loss, psnr))

    return results
# def main(args):
if __name__ == '__main__':
    args = get_args()
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

    train_sigma = snr_db2sigma(args.train_snr)
    val_sigma = snr_db2sigma(args.val_snr)
    test_snr = np.arange(-60, 60, 10)
    test_sigma = np.array([snr_db2sigma(snr) for snr in test_snr])
    if not args.scale_outputs:       
        eq_train_sigma = train_sigma / np.sqrt(args.power)  
        eq_val_sigma = val_sigma / np.sqrt(args.power)
        eq_test_sigma = test_sigma / np.sqrt(args.power)
        scale_power = 1
    else:
        eq_train_sigma = train_sigma
        eq_val_sigma = val_sigma
        eq_test_sigma = test_sigma
        scale_power = args.power

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

    model = ButterflyNetwork(args.input_size, args.hidden_size, args.output_size, eq_train_sigma, scale_power).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if not args.test:
        try:
            with tqdm(range(args.epochs), desc="Epochs") as epoch_progress:
                if args.dropout:
                    dropped_list = ['AE', 'BF']
                else:
                    dropped_list = []
                for epoch in epoch_progress:
                    train_loss, train_psnr = train(model, train_loader, optimizer, device, dropped_list)
                    results = test(model, test_loader, np.array([eq_val_sigma]), device)
                    val_snrs, test_losses, test_psnrs = zip(*results)
                    val_snr = val_snrs[0]
                    test_loss = test_losses[0]
                    test_psnr = test_psnrs[0]

                    epoch_progress.set_postfix({"Val SNR": val_snr, "Test Loss": test_loss, "Test PSNR": test_psnr})

                    result_str = f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}'
                    for snr, loss, psnr in results:
                        result_str += f', Test (SNR {snr:.2f}dB) Loss: {loss:.4f}, PSNR: {psnr:.4f}'

                    epoch_progress.write(result_str)

                    # save best model
                    if epoch == 0:
                        best_loss = test_loss
                        torch.save(model.state_dict(), model_path)
                    else:
                        if test_loss < best_loss:
                            best_loss = test_loss
                            torch.save(model.state_dict(), model_path)

            print(f"Best Test Loss: {best_loss:.4f}")
            print(f"Model saved at {model_path}")
        except KeyboardInterrupt:
            print("Interrupted by user")
            print(f"Model saved at {model_path}")
            torch.save(model.state_dict(), model_path)

    # testing
    print(f"Testing")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        # snr_range = np.arange(0, 60, 10)
        result = test(model, test_loader, eq_test_sigma, device)
        snr_range, test_losses, test_psnrs = zip(*result)
        test_losses_str = ', '.join([f"{loss:.4f}" for loss in test_losses])
        test_psnrs_str = ', '.join([f"{psnr:.4f}" for psnr in test_psnrs])
        print(f"Loaded model:\n SNR range: {snr_range}\n Test Loss: {test_losses_str}\n Test PSNR: {test_psnrs_str}")

        result = test_dropped(model, test_loader, eq_test_sigma, device, ['AE'])
        snr_range, test_losses, test_psnrs = zip(*result)
        test_losses_str = ', '.join([f"{loss:.4f}" for loss in test_losses])
        test_psnrs_str = ', '.join([f"{psnr:.4f}" for psnr in test_psnrs])
        print(f"Loaded model:\n SNR range: {snr_range}\n Test Loss: {test_losses_str}\n Test PSNR: {test_psnrs_str}")
    else:
        print(f"No model found at {model_path}")


    
# if __name__ == '__main__':
#     args = get_args()
#     main(args)
