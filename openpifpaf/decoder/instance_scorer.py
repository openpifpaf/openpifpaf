import torch


class InstanceScorer(torch.nn.Module):
    def __init__(self, in_features=35):
        super(InstanceScorer, self).__init__()
        self.compute_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.compute_layers(x - 0.5)

    def from_annotation(self, ann):
        v = torch.tensor([ann.scale()] +
                         ann.data[:, 2].tolist() +
                         ann.joint_scales.tolist()).float()
        with torch.no_grad():
            return float(self.forward(v).item())
