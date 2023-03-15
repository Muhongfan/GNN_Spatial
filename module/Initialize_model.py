import torch
import torch.optim as optim
from utils import set_seed

def initialize_model(model, learning_rate, device):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # model design
    set_seed(42)
    # net = GCN(input_dim=6 * 1, hidden_dim=6, output_dim=1)
    net = model
    net = net.to(device)
    
    # loss and optimizer
    criterion = torch.nn.MSELoss()

    # Create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # # Total number of training steps
    # total_steps = len(test_loader) * epochs
    # 
    # train
    #Epoch = 2

    # patience = 0.2 * Epoch
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    # 

    return net, criterion, optimizer, scheduler