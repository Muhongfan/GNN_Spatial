import torch.nn.functional as F
import time
import torch
import numpy as np
import h5py

from utils import visualize_Result, plot_loss, save_feats
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import compute_performance


def test(model, device, criterion, test_dataloader, loss_train_plt, loss_val_plt, path, num_nodes, slot):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # the test time.
    model.eval()

    # Tracking variables
    test_loss = []
    total_loss = 0.0

    MAE, MAPE, RMSE, ACC, R2, Explained_VAR = [], [], [], [], [], []
    # PEMS04
    # Diff num of nodes
    Target = np.zeros([num_nodes, 1, 1])  # [N, T, D]
    # LOS
    Predict = np.zeros_like(Target)

    # For each batch in our test set...
    for data in test_dataloader:
        # Compute prediction
        test_flow_x = data["flow_x"][0]
        # print("test data:,", test_flow_x)
        with torch.no_grad():
            predict_value = model(data, device).to(torch.device("cpu"))
        # save_feats(predict_value, path)
        # Compute loss
        loss = criterion(predict_value, data["flow_y"])
        total_loss += loss.item()
        test_loss.append(loss.item())

        # concatenate each time slot
        predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
        target_value = data["flow_y"].transpose(0, 2).squeeze(0)

        performance, data_save = compute_performance(predict_value, target_value, test_dataloader)

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

        # print("Test loss: {:02.4f}".format(1000 * total_loss / len(test_dataloader)))
    # print(test_loss)
    test_losses = np.mean(test_loss)
    # print(test_losses)

    # te_loss = np.average(test_loss)
    accurancy = np.mean(ACC)
    # print("difference of test loss: test losses{:2.2f}, test loss{:2.2f}, total loss{:2.2f}", total_loss,
    #       test_losses, te_loss)
    print(
        "Performance: MAE {:2.4f}, MAPE {:2.4f}, RMSE {:2.4f},  ACC{:2.5f}, R2 {:2.5f}, Explained_VAR {:2.5f}, Test_loss{:2.7f}".format(
            np.mean(MAE), np.mean(MAPE), np.mean(RMSE), accurancy, np.mean(R2), np.mean(Explained_VAR),
            test_losses))

    name = model.__class__.__name__
    # #timeslot = 30
    # #name = name + str(timeslot)
    # path = "./Losloop/Model_" +str(slot) +'/' + name +"/"
    # if not os.path.isdir(path):
    #     os.makedirs(path)

    stats_path = path + "Stats/"
    if not os.path.isdir(stats_path):
        os.makedirs(stats_path)
    file_stats = stats_path + "stats.npz"
    np.savez(file_stats,
             k_a=[np.mean(MAE), np.mean(MAPE), np.mean(RMSE), np.mean(ACC), np.mean(R2), np.mean(Explained_VAR)])

    # c = np.load('ab.npz')

    Predict = np.delete(Predict, 0, axis=1)
    # print(Predict)
    Target = np.delete(Target, 0, axis=1)
    # print(Target)

    # print(str(net))
    # print(name)

    result_file = name + ".h5"

    # if not os.path.isdir(path):
    #     os.makedirs(path)
    save_path = path + result_file

    file_obj = h5py.File(save_path, "w")

    file_obj["predict"] = Predict
    file_obj["target"] = Target
    # print(loss_train_plt)
    # print(loss_val_plt)
    plot_loss(path=path,
              visualize_file=name,
              train_loss=loss_train_plt,
              val_loss=loss_val_plt
              )

    visualize_Result(path=path,
                     h5_file=result_file,
                     nodes_id=8,
                     time_se=[0, 24 * 12 * 2],
                     visualize_file=name)

    return accurancy




    



