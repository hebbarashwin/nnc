import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    parser.add_argument('-ni', '--num_inputs', type=int, default=2, help='graph with 2 or 4 inputs?')
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
    parser.add_argument('--dropout', type=float, default=0., help='Add dropout during training')
    parser.add_argument('--test', action='store_true', help='Only test the model')
    parser.add_argument('--pruned', action='store_true', help='Only test the model')
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

def train(model, dataloader, optimizer, device, dropped_list = None, dropout = 0.):
    model.train()
    running_loss = 0.0

    progress = tqdm(dataloader, leave=False, desc="Batches")
    for images, labels in progress:
        optimizer.zero_grad()
        
        if len(model.graph_inputs) == 2:
            x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
            x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
            x = [x1, x2]
        elif len(model.graph_inputs) == 4:
            x1 = images[:, :, :14, :14].reshape(images.size(0), -1).to(device)
            x2 = images[:, :, 14:, :14].reshape(images.size(0), -1).to(device)
            x3 = images[:, :, :14, 14:].reshape(images.size(0), -1).to(device)
            x4 = images[:, :, 14:, 14:].reshape(images.size(0), -1).to(device)
            x = [x1, x2, x3, x4]
        
        output_list = model(x, dropout_probability = dropout, dropped = dropped_list)
        
        target = images.view(images.size(0), -1).to(device)
        
        # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
        # loss = nn.BCEWithLogitsLoss()(y, target)
        # loss = nn.MSELoss()(y, target)
        loss = torch.mean(torch.stack([nn.MSELoss()(y, target) for y in output_list]))
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
                # x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                # x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                
                # if not quantize:
                #     y1, y2, y_list = model(x1, x2, noise_std_dev=sigma)
                # else:
                #     y1, y2, y_list = model.run_quantized(num_bits, x1, x2, noise_std_dev=sigma)
                
                # y = torch.cat((y1, y2), dim=1)
                # target = torch.cat((x1, x2), dim=1)
                
                # # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # # loss = nn.BCEWithLogitsLoss()(y, target)
                # loss = nn.MSELoss()(y, target)

                if len(model.graph_inputs) == 2:
                    x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                    x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                    x = [x1, x2]
                elif len(model.graph_inputs) == 4:
                    x1 = images[:, :, :14, :14].reshape(images.size(0), -1).to(device)
                    x2 = images[:, :, 14:, :14].reshape(images.size(0), -1).to(device)
                    x3 = images[:, :, :14, 14:].reshape(images.size(0), -1).to(device)
                    x4 = images[:, :, 14:, 14:].reshape(images.size(0), -1).to(device)
                    x = [x1, x2, x3, x4]
                
                output_list = model(x, noise_std_dev = sigma)#, dropped = dropped_list)
                
                target = images.view(images.size(0), -1).to(device)
                
                # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # loss = nn.BCEWithLogitsLoss()(y, target)
                # loss = nn.MSELoss()(y, target)
                loss = torch.mean(torch.stack([nn.MSELoss()(y, target) for y in output_list]))

                running_loss += loss.item()
            running_loss = running_loss / len(dataloader)
            psnr = 10*np.log10(torch.max(images)**2 / running_loss).item()
            results.append((snr, running_loss, psnr))

    return results

def test_dropped(model, dataloader, test_sigma_range, device, dropped = [] , quantize = False, num_bits = 8):
    
    results = []
    model.eval()    
    with torch.no_grad():
        for sigma in test_sigma_range:
            snr = snr_sigma2db(sigma)
            running_loss = 0.0
            for images, labels in dataloader:
                # x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                # x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                
                # if not quantize:
                #     y1, y2, y_list = model(x1, x2, noise_std_dev=sigma)
                # else:
                #     y1, y2, y_list = model.run_quantized(num_bits, x1, x2, noise_std_dev=sigma)
                
                # y = torch.cat((y1, y2), dim=1)
                # target = torch.cat((x1, x2), dim=1)
                
                # # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # # loss = nn.BCEWithLogitsLoss()(y, target)
                # loss = nn.MSELoss()(y, target)

                if len(model.graph_inputs) == 2:
                    x1 = images[:, :, :14, :].view(images.size(0), -1).to(device)
                    x2 = images[:, :, 14:, :].view(images.size(0), -1).to(device)
                    x = [x1, x2]
                elif len(model.graph_inputs) == 4:
                    x1 = images[:, :, :14, :14].reshape(images.size(0), -1).to(device)
                    x2 = images[:, :, 14:, :14].reshape(images.size(0), -1).to(device)
                    x3 = images[:, :, :14, 14:].reshape(images.size(0), -1).to(device)
                    x4 = images[:, :, 14:, 14:].reshape(images.size(0), -1).to(device)
                    x = [x1, x2, x3, x4]
                
                output_list = model(x, noise_std_dev = sigma, dropped = dropped, dropout_probability = 1)#, dropped = dropped_list)
                
                target = images.view(images.size(0), -1).to(device)
                
                # loss = custom_loss(y, target, adjacency_matrix, lambda_matrix, y_list)
                # loss = nn.BCEWithLogitsLoss()(y, target)
                # loss = nn.MSELoss()(y, target)
                loss = torch.mean(torch.stack([nn.MSELoss()(y, target) for y in output_list]))

                running_loss += loss.item()
            running_loss = running_loss / len(dataloader)
            psnr = 10*np.log10(torch.max(images)**2 / running_loss).item()
            results.append((snr, running_loss, psnr))

    return results


import torch
import torch.nn as nn

class Node:
    def __init__(self, name, output_nodes):
        self.name = name
        self.output_nodes = output_nodes
        self.input_nodes = []

class GeneralDAG(nn.Module):
    def __init__(self, nodes_dict, input_size, hidden_size, output_size, noise_std_dev, scale_power=1):
        super(GeneralDAG, self).__init__()
        self.nodes = {}
        self.nn_modules = nn.ModuleDict()
        self.noise_std_dev = noise_std_dev
        self.scale_power = scale_power
        self.edges = []
        # Create Node objects
        for name, output_nodes in nodes_dict.items():
            self.nodes[name] = Node(name, output_nodes)
            self.edges.extend([(name, onn) for onn in output_nodes])
        
        # Add inputs to Node objects
        for node in self.nodes.values():
            for output_node in node.output_nodes:
                if output_node in self.nodes:
                    self.nodes[output_node].input_nodes.append(node.name)

        self.graph_inputs = self.get_graph_inputs()
        self.graph_inputs.sort()
        self.graph_outputs = self.get_graph_outputs()
        self.graph_outputs.sort()

        for node in self.nodes.values():
            num_inputs = len(node.input_nodes)
            num_outputs = len(node.output_nodes) 
            if num_inputs == 0:
                i_size = input_size 
            else:
                i_size = num_inputs * output_size

            if num_outputs == 0:
                o_size = len(self.graph_inputs)*input_size 
            else:
                o_size = output_size 
            self.nn_modules[str(node.name)] = SimpleFullyConnected(i_size, hidden_size, o_size)

    def topological_sort(self):
        result = []
        visited = set()

        def visit(node):
            if node.name not in visited:
                visited.add(node.name)
                for output_node in node.output_nodes:
                    if output_node in self.nodes:
                        visit(self.nodes[output_node])
                result.append(node.name)

        for node in self.nodes.values():
            visit(node)
        
        return result[::-1]

    def get_graph_inputs(self):
        return [node.name for node in self.nodes.values() if not node.input_nodes]

    def get_graph_outputs(self):
        return [node.name for node in self.nodes.values() if not node.output_nodes] 

    def link_dropout(self, x, p):
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

    def add_noise(self, x, noise_std_dev):
        # Add Gaussian noise to the input tensor
        noise = torch.randn_like(x) * noise_std_dev
        return x + noise

    def forward(self, x, noise_std_dev = None, dropped=None, dropout_probability = 0.):

        if noise_std_dev is None:
            noise_std_dev = float(self.noise_std_dev)  # Ensure that noise_std_dev is a scalar (float)

        if dropped is None:
            dropped = []#self.edges 

        
        assert 0 <= dropout_probability <= 1

        # print(dropped, dropout_probability)
        forward_outputs = {}
        assert len(x) == len(self.graph_inputs)
        for i, node_name in enumerate(self.graph_inputs) :
            forward_outputs[node_name] = self.nn_modules[str(node_name)](x[i])

        node_order = self.topological_sort()
        # I want to traverse the graph nodes only after all their parents have been traversed.
        node_order = [n for n in node_order if n not in self.graph_inputs]
        for node_name in node_order:
            node = self.nodes[node_name]
            
            # NN_input = torch.cat([self.add_noise(forward_outputs[input_node], noise_std_dev) for input_node in node.input_nodes], dim = -1)
            NN_input = torch.cat([self.add_noise(self.link_dropout(forward_outputs[input_node], dropout_probability if (input_node, node_name) not in dropped else 1), noise_std_dev) for input_node in node.input_nodes], dim=-1)
            forward_outputs[node_name] = self.nn_modules[str(node_name)](NN_input) * self.scale_power

        return [forward_outputs[node_name] for node_name in self.graph_outputs]



# def main(args):
if __name__ == '__main__':
    args = get_args()
    model_path = os.path.join('g4models', args.model_path, 'model.pt')
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Model path : {model_path}")

    # # define graph nodes : 4 inputs
    if args.num_inputs == 4:
        if args.pruned:
            nodes = {}
            nodes[0] = [5]
            nodes[1] = [4,5]
            nodes[2] = [5]
            nodes[3] = [6,9]
            nodes[4] = [7]
            nodes[5] = [8,9]
            nodes[6] = [7,8,9]
            nodes[7] = [10,11]
            nodes[8] = [10]
            nodes[9] = [10,11]
            nodes[10] = []
            nodes[11] = []
        else:
            nodes = {}
            nodes[0] = [4,5]
            nodes[1] = [4,5]
            nodes[2] = [5,6]
            nodes[3] = [6,9]
            nodes[4] = [7]
            nodes[5] = [7,8,9]
            nodes[6] = [7,8,9]
            nodes[7] = [10,11]
            nodes[8] = [10]
            nodes[9] = [10,11]
            nodes[10] = []
            nodes[11] = []

    # define graph nodes : 2 inputs - butterfly
    elif args.num_inputs == 2:
        nodes = {}
        nodes[0] = [2,4]
        nodes[1] = [2,5]
        nodes[2] = [3]
        nodes[3] = [4,5]
        nodes[4] = []
        nodes[5] = []

    # change device to cuda or cpu .. mps is for mac-m1 gpu
    if args.device == 'cpu':
        device = torch.device("cpu") 
    elif 'cuda' in args.device:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = (28*28)//args.num_inputs 
    model = GeneralDAG(nodes, input_size, args.hidden_size, args.output_size, eq_train_sigma, scale_power).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if not args.test:
        try:
            with tqdm(range(args.epochs), desc="Epochs") as epoch_progress:
                # if args.dropout:
                #     dropped_list = ['AE', 'BF']
                # else:
                #     dropped_list = []
                dropped_list = None # If only specific edges, pass list of input-output tuples.
                for epoch in epoch_progress:
                    train_loss, train_psnr = train(model, train_loader, optimizer, device, dropped_list, args.dropout)
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

        result = test_dropped(model, test_loader, eq_test_sigma, device, [(5,7), (2,6), (0,4)])
        snr_range, test_losses, test_psnrs = zip(*result)
        test_losses_str = ', '.join([f"{loss:.4f}" for loss in test_losses])
        test_psnrs_str = ', '.join([f"{psnr:.4f}" for psnr in test_psnrs])
        print(f"Loaded model:\n SNR range: {snr_range}\n Test Loss: {test_losses_str}\n Test PSNR: {test_psnrs_str}")
    else:
        print(f"No model found at {model_path}")


    
# if __name__ == '__main__':
#     args = get_args()
#     main(args)
