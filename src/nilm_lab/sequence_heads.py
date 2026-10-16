"""Predicted state probabilities -> activity; no teacher forcing or hidden carry."""
from torch import nn


class SequenceActivityHead(nn.Module):
    def __init__(self, classes, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(classes, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, logits):
        probabilities = logits.softmax(dim=1).transpose(1, 2)
        # None initial state on every call/window; gradients flow through softmax.
        h, _ = self.gru(probabilities)
        return self.output(h).squeeze(-1)


def direct_activity_head(dim, hidden_size=32):
    return nn.Sequential(nn.Conv1d(dim, hidden_size, 1), nn.ReLU(), nn.Conv1d(hidden_size, 1, 1))


class PointActivityHead(nn.Module):
    """Same probability input as GRU; no additional temporal receptive field."""
    def __init__(self, classes, hidden_size=32):
        super().__init__()
        budget = 3 * hidden_size * (classes + hidden_size + 2) + hidden_size + 1
        width = max(1, round((budget - 1) / (classes + 2)))
        self.network = direct_activity_head(classes, width)

    def forward(self, logits):
        return self.network(logits.softmax(1)).squeeze(1)


class FeatureActivityHead(nn.Module):
    def __init__(self, dim, classes, hidden_size=32):
        super().__init__()
        self.project = nn.Conv1d(dim, classes, 1)
        self.gru = nn.GRU(classes, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features):
        hidden, _ = self.gru(self.project(features).transpose(1, 2))
        return self.output(hidden).squeeze(-1)
