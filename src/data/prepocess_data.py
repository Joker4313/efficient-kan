import numpy as np
import pandas as pd


def add_label_noise(y_train: pd.DataFrame, noise_level):
    num_swap_labels = int(np.round(len(y_train) * noise_level)) # calculate the number of labels to swap
    swap_indices = np.random.choice(y_train.index, num_swap_labels * 2, replace=False) # randomly select indices to swap

    for i in range(0, len(swap_indices), 2):
        y_train.loc[swap_indices[i]], y_train.loc[swap_indices[i + 1]] = (
            y_train.loc[swap_indices[i + 1]].copy(),
            y_train.loc[swap_indices[i]].copy(),
        )

    return y_train
