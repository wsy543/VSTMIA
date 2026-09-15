import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def path_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")




def load_npz_data(data_name):
    loaded_data = np.load(data_name)

    loaded_split_data_list = []
    loaded_split_labels_list = []

    client_indices = set()
    for key in loaded_data.files:
        if key.startswith('client_') and key.endswith('_data'):
            try:
                idx_str = key[len('client_'):-len('_data')]
                client_idx = int(idx_str)
                client_indices.add(client_idx)
            except ValueError:
                pass

    num_clients = len(client_indices)
    sorted_client_indices = sorted(list(client_indices))

    if num_clients == 0:
        loaded_data.close()
        return [], []

    for client_idx in sorted_client_indices:
         data_key = f'client_{client_idx}_data'
         labels_key = f'client_{client_idx}_labels'

         loaded_split_data_list.append(loaded_data[data_key])
         loaded_split_labels_list.append(loaded_data[labels_key])

    loaded_data.close()

    return loaded_split_data_list, loaded_split_labels_list
