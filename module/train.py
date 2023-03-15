import time
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import compute_performance, set_seed
from Earlystopping import EarlyStopping
from utils import plot_loss
def train(model, device, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, num_nodes, path, evaluation=True):
    """Train the BertClassifier model.
    """
    best_loss = float('inf')
    # set_seed(42)

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} |  {'Elapsed':^9}")
    print("-" * 70)
    loss_train_plt = []
    loss_val_plt = []

    patience = 0.1 * epochs
    early_stopping = EarlyStopping(path, patience = patience, verbose = True)


    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Measure the elapsed time of each epoch
        t0_epoch = time.time()

        # Reset tracking variables at the beginning of each epoch
        epoch_loss = 0
        train_losses = []


        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for data in train_dataloader:
            # torch.cuda.empty_cache()
            # Load batch to GPU
            #train_flow_x = data["flow_x"][0]
            #print("train data:,", train_flow_x)
            data_y = data["flow_y"]
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            predict = model(data, device).to(torch.device("cpu"))
            # p2, p3, p4, p5 = model(data, device)


            # Compute loss and accumulate the loss values
            loss = criterion(predict, data_y)
            epoch_loss = loss.item()
            train_losses.append(epoch_loss)

            #
            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()
            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = epoch_loss / len(train_dataloader)
        loss_train_plt.append(10 * epoch_loss / len(train_dataloader) / 64)
        #print(len(loss_train_plt))

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss = evaluate(model, device, criterion, val_dataloader, num_nodes)
            loss_val_plt.append(10 * val_loss / len(val_dataloader) / 64)
            if val_loss < best_loss:
                best_loss = val_loss

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch


        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        

        print(
            f"{epoch_i + 1:^7} | {'-':^7} | {loss_train_plt[-1]:^12.6f} | {val_loss:^10.6f} |  {time_elapsed:^9.2f}")

        
    # print(f"Training complete! Best loss: {best_loss:^12.6f}.")
    # print(loss_train_plt)
    # print(loss_val_plt)
    
    
    return loss_train_plt,loss_val_plt,best_loss


def evaluate(model, device, criterion, val_dataloader, num_nodes):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_loss = []
    total_loss = 0.0

    MAE, MAPE, RMSE, ACC, R2, Explained_VAR = [], [], [], [], [], []
    # PEMS04
    # Diff num of nodes
    Target = np.zeros([num_nodes, 1, 1])  # [N, T, D]
    # LOS
    Predict = np.zeros_like(Target)
    
    
    # For each batch in our validation set...
    for data in val_dataloader:
        # Compute prediction
        #print("valid data:," ,data["flow_x"][0])
        with torch.no_grad():
            predict_value = model(data, device).to(torch.device("cpu"))


        # Compute loss
        loss = criterion(predict_value, data["flow_y"])
        total_loss += loss.item()
        val_loss.append(loss.item())

        # concatenate each time slot
        predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
        target_value = data["flow_y"].transpose(0, 2).squeeze(0)
        
        performance, data_save = compute_performance(predict_value, target_value, val_dataloader)

        Predict = np.concatenate([Predict, data_save[0]], axis=1)
        # print(len(Predict))
        Target = np.concatenate([Target, data_save[1]], axis=1)
        # (Target)

        MAE.append(performance[0])
        MAPE.append(performance[1])
        RMSE.append(performance[2])
        ACC.append(performance[3])
        R2.append(performance[4])
        Explained_VAR.append(performance[5])

        #print("Test loss: {:02.4f}".format(1000 * total_loss / len(val_dataloader)))

    valid_losses = np.mean(val_loss)
    val = int(valid_losses*1000000)/1000000
    # print(val)



    #valid_loss = np.average(val_loss)

    # print("Difference of valid loss: valid losses{:2.2f}, valid loss{:2.2f}, total loss{:2.2f}".format( total_loss, valid_losses, valid_loss))
    # print("Performance: MAE {:2.2f}, MAPE {:2.2f}, RMSE {:2.2f},  ACC{:2.2f}, R2 {:2.2f}, Explained_VAR {:2.2f}, Test_loss{:2.2f}".format(
    #     np.mean(MAE), np.mean(MAPE), np.mean(RMSE), np.mean(ACC), np.mean(R2), np.mean(Explained_VAR), valid_losses))

    return val