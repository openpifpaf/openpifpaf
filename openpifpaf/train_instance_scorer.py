import json
import random

import pysparkling
import torch

from .decoder.instance_scorer import InstanceScorer
from . import show

DATA_FILE = ('outputs/resnet101block5-pif-paf-edge401-190412-151013.pkl'
             '.decodertraindata-edge641-samples0.json')


def plot_training_data(train_data, val_data, entry=0, entryname=None):
    train_x, train_y = train_data
    val_x, val_y = val_data
    with show.canvas() as ax:
        ax.hist([xx[entry] for xx in train_x[train_y[:, 0] == 1]],
                bins=50, alpha=0.3, density=True, color='navy', label='train true')
        ax.hist([xx[entry] for xx in train_x[train_y[:, 0] == 0]],
                bins=50, alpha=0.3, density=True, color='orange', label='train false')

        ax.hist([xx[entry] for xx in val_x[val_y[:, 0] == 1]],
                histtype='step', bins=50, density=True, color='navy', label='val true')
        ax.hist([xx[entry] for xx in val_x[val_y[:, 0] == 0]],
                histtype='step', bins=50, density=True, color='orange', label='val false')

        if entryname:
            ax.set_xlabel(entryname)
        ax.legend()


def train_val_split_score(data, train_fraction=0.6, balance=True):
    xy_list = data.map(lambda d: ([d['score']], [float(d['target'])])).collect()

    if balance:
        n_true = sum(1 for x, y in xy_list if y[0] == 1.0)
        n_false = sum(1 for x, y in xy_list if y[0] == 0.0)
        p_true = min(1.0, n_false / n_true)
        p_false = min(1.0, n_true / n_false)
        xy_list = [(x, y) for x, y in xy_list
                   if random.random() < (p_true if y[0] == 1.0 else p_false)]

    n_train = int(train_fraction * len(xy_list))

    return (
        (torch.tensor([x for x, _ in xy_list[:n_train]]),
         torch.tensor([y for _, y in xy_list[:n_train]])),
        (torch.tensor([x for x, _ in xy_list[n_train:]]),
         torch.tensor([y for _, y in xy_list[n_train:]])),
    )


def train_val_split_keypointscores(data, train_fraction=0.6):
    xy_list = (
        data
        .map(lambda d: ([d['score']] + [xyv[2] for xyv in d['keypoints']] + d['joint_scales'],
                        [float(d['target'])]))
        .collect()
    )
    n_train = int(train_fraction * len(xy_list))

    return (
        (torch.tensor([x for x, _ in xy_list[:n_train]]),
         torch.tensor([y for _, y in xy_list[:n_train]])),
        (torch.tensor([x for x, _ in xy_list[n_train:]]),
         torch.tensor([y for _, y in xy_list[n_train:]])),
    )


def train_epoch(model, loader, optimizer):
    epoch_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()

        y_hat = model(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)

def val_epoch(model, loader):
    epoch_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            y_hat = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            epoch_loss += float(loss.item())

    return epoch_loss / len(loader)


def main():
    sc = pysparkling.Context()
    data = sc.textFile(DATA_FILE).map(json.loads).cache()

    train_data_score, val_data_score = train_val_split_score(data)
    plot_training_data(train_data_score, val_data_score, entryname='score')

    train_data, val_data = train_val_split_keypointscores(data)

    model = InstanceScorer()

    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(*val_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    for epoch_i in range(100):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = val_epoch(model, val_loader)
        print(epoch_i, train_loss, val_loss)

    with torch.no_grad():
        post_train_data = (model(train_data[0]), train_data[1])
        post_val_data = (model(val_data[0]), val_data[1])
    plot_training_data(post_train_data, post_val_data, entryname='optimized score')
    torch.save(model, 'instance_scorer.pkl')


if __name__ == '__main__':
    main()
