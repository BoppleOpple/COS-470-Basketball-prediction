import random
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import numpy as np
class BasketBallDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame):
        self.data_names = data_frame.columns.tolist()
        self.labels = torch.from_numpy(data_frame.values[:, :1])
        self.data = torch.from_numpy(data_frame.values[:, 1:])

        print(self.labels.shape)
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def process(input_path):
    data = pd.read_csv(input_path)

    # determine which years are testing
    year_range = (data['YEAR'].min(), data['YEAR'].max())
    test_years = random.sample(range(*year_range), 2);

    # team_names = data['TEAM'].unique()
    # test_teams = random.choices(team_names, k=int(len(team_names) * 0.15))

    # print("test teams: ", test_teams)


    # just learned about the tilde operator, seems fun
    train_data = data[~data['YEAR'].isin(test_years)]
    test_data = data[data['YEAR'].isin(test_years)]

    # train_data = data[~data['TEAM'].isin(test_teams)]
    # test_data = data[data['TEAM'].isin(test_teams)] 

    # # sample 100% of the data (shuffle)
    # data = data.sample(frac=1).reset_index(drop=True)

    # split_index = int(len(data) * 0.85)

    # train_data = data[:split_index]
    # test_data = data[split_index:]

    return BasketBallDataset(train_data.iloc[:, :13]), BasketBallDataset(test_data.iloc[:, :13])

def visualize(dataset):
    # messing around with the data, maybe i'll find something helpful
    x_index = 6
    y_index = 0

    x_label = dataset.data_names[x_index]
    y_label = dataset.data_names[y_index]

    plt.figure()
    plt.scatter(dataset.data[:, x_index], dataset.data[:, y_index])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please provide the following arguments:")
        print("python preprocess.py <input_path> [output_path]")
        print("if output_path is not provided, the output will be saved to ./data/preprocessed.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./data/preprocessed.csv"
    
    train_data, test_data = process(input_path)

    print(f"train data length: {len(train_data)} ({len(train_data) / (len(train_data) + len(test_data)) * 100:.2f}%)")
    print(f"test data length: {len(test_data)} ({len(test_data) / (len(train_data) + len(test_data)) * 100:.2f}%)")
    print(train_data[1])

    visualize(train_data)
    visualize(test_data)
