
from torch.optim.lr_scheduler import OneCycleLR


if __name__ == "__main__":
    from torch.optim import SGD
    from torch.nn import Linear
    import matplotlib.pyplot as plt

    epochs = 1_000
    max_lr = 0.1

    nn = Linear(1, 1)
    optimizer = SGD(nn.parameters(), lr=1.)
    scheduler = OneCycleLR(optimizer,
                           max_lr=max_lr,
                           epochs=epochs,
                           steps_per_epoch=1)
    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        scheduler.step()
        # print(f' epoch: {epoch}, lr:{ scheduler.get_last_lr()}')
        lrs.append(scheduler.get_last_lr()[0])

    plt.title("OneCycleLR")
    plt.plot(lrs)
    for i in range(11):
        plt.axvline(x=i*epochs/10, color="grey", linestyle='--')
    plt.xticks([i*epochs/10 for i in range(11)])
    plt.xlabel("Epochs")

    for i in range(11):
        plt.axhline(y=max_lr/10*i, color="grey", linestyle='-', alpha=0.5)
    plt.yticks([max_lr/10*i for i in range(11)])
    plt.ylabel("Learning Rate")

    plt.show()
