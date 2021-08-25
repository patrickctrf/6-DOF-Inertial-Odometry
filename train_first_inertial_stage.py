import torch
from keras.models import load_model
from matplotlib import pyplot as plt
from numpy import arange, random, array
from pandas import read_csv
from tqdm import tqdm
import numpy as np

from mydatasets import *
from ptk.timeseries import *


def experiment():
    """
Runs the experiment itself.

    :return: Trained model.
    """

    model = load_model("6dofio_euroc.hdf5")

    # ===========PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]============
    room2_tum_dataset = AsymetricalTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv",
                                                     min_window_size=30, max_window_size=31, shuffle=False, device=device, convert_first=True)

    euroc_v2_dataset = AsymetricalTimeseriesDataset(x_csv_path="V1_01_easy/mav0/imu0/data.csv", y_csv_path="V1_01_easy/mav0/vicon0/data.csv",
                                                    min_window_size=100, max_window_size=101, shuffle=False, device=device, convert_first=True)

    dados_de_entrada_imu = read_csv("V1_01_easy/mav0/imu0/data.csv").to_numpy()[:, 1:]
    dados_de_saida = read_csv("V1_01_easy/mav0/vicon0/data.csv").to_numpy()[:, 1:]

    predict = []
    for i in tqdm(range(0, dados_de_entrada_imu.shape[0] - 200, 10)):
        predict.append(
            np.hstack(
                model.predict([
                    dados_de_entrada_imu[i:i + 200, :3].reshape(-1, 200, 3),
                    dados_de_entrada_imu[i:i + 200, 3:].reshape(-1, 200, 3)],
                    batch_size=1, verbose=1)
            )[0]
        )

    predict = array(predict)

    dimensoes = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    for i, dim_name in enumerate(dimensoes):
        plt.close()
        plt.plot(range(0, dados_de_saida.shape[0], dados_de_saida.shape[0] // predict.shape[0])[:predict.shape[0]], predict[:, i])
        plt.plot(range(dados_de_saida.shape[0]), dados_de_saida[:, i])
        plt.legend(['predict', 'reference'], loc='upper right')
        plt.savefig(dim_name + ".png", dpi=200)
        plt.show()

    # ===========FIM-DE-PREDICAO-["px", "py", "pz", "qw", "qx", "qy", "qz"]=====

    print(model)

    return model


if __name__ == '__main__':

    # plot_csv()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("Usando GPU")
    else:
        dev = "cpu"
        print("Usando CPU")
    device = torch.device(dev)

    experiment()
