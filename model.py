import torch
import config
import numpy as np
import matplotlib.pyplot as plt

class BasketBallModel(torch.nn.Module):
    def __init__(self):
        super(BasketBallModel, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(12, 124, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(124, 32, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1, dtype=torch.float64)
        )

    def forward(self, x):
        return self.fc(x)


def train(model, train_data, test_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # print(train_data.data.dtype, test_data.data.dtype)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=True)

    train_losses = np.zeros(config.NUM_EPOCHS)
    test_losses = np.zeros(config.NUM_EPOCHS)


    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses[epoch] += loss.item() / len(train_loader)
        # scheduler.step()

        model.eval()
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_losses[epoch] += loss.item() / len(test_loader)

            # print("---")
            # print(outputs)
            # print(labels)
            # print((outputs.round() == labels).sum().item())
            # print("---")


        print(f"Epoch {epoch}, Batch size: {config.BATCH_SIZE}, Train Loss: {train_losses[epoch]}, Test Loss: {test_losses[epoch]}")
    

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.savefig(f"L1_loss_{config.BATCH_SIZE}batch_size_{config.NUM_EPOCHS}epochs_{optimizer.param_groups[0]['lr']}lr.png")

    plt.close() 

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"log_L1_loss_{config.BATCH_SIZE}batch_size_{config.NUM_EPOCHS}epochs_{optimizer.param_groups[0]['lr']}lr.png")
    plt.close() 

def test(model, test_data):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    accuracy_fn = torch.nn.L1Loss()

    correct = 0
    cumulative_loss = 0

    model.eval()

    for inputs, labels in test_loader:
        outputs = model(inputs)

        print("number of batches: ", len(test_loader))

        correct += (outputs.round() == labels).sum().item()
        cumulative_loss += accuracy_fn(outputs, labels).item() / len(test_loader)

        residuals = outputs - labels
        residuals = residuals.detach().numpy()

        if (len(test_loader) == 1):
            plt.hist(residuals, bins=100)
            plt.savefig(f"residuals.png")
            plt.close()
    

    print(f"Accuracy: {correct / len(test_data)}")

    print(f"L1 Loss: {cumulative_loss}")
