from torch.utils.data import DataLoader

from data import Data
from lstm import LSTMXOR
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    model = LSTMXOR(1, 2, 1)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loader = DataLoader(
                    Data(sample_size=100000),
                    batch_size=8
                    )
    model.train()

    print("RUNNING TRAINING...")

    for epoch in range(8):

        print("EPOCH:", epoch+1)

        for step, (features, labels) in enumerate(train_loader):
            # pass
            outputs = model(features)
            loss = loss_function(outputs, labels)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = ((outputs > 0.5) == (labels > 0.5)).type(torch.FloatTensor).mean()
            acc_dif = abs(accuracy.item() - 1.0)

            if step % 200 == 0:
                print(" -- For step:", step, "accuracy is: {:.3f}".format(accuracy))

                if acc_dif < 0.0001:
                    print("EARLY STOPPING")
                    return 0

    return 0


main()
