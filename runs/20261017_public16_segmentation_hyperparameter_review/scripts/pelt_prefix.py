"""Official ruptures PELT with cached sums of its unchanged official RBF Gram."""
import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
from ruptures.costs import CostRbf
from ruptures.exceptions import NotEnoughPoints
from models.time_segmentation.comparison import Unsupported


class PrefixRbf(BaseCost):
    model = 'rbf'
    min_size = 1

    def fit(self, signal):
        if getattr(self, '_input', None) is signal:
            return self
        self._input = signal
        self.official = CostRbf().fit(signal)
        self.signal = self.official.signal
        gram = self.official.gram
        self.gamma = self.official.gamma
        self.prefix = np.zeros((len(gram)+1, len(gram)+1), dtype=np.float64)
        self.prefix[1:, 1:] = gram
        np.cumsum(self.prefix[1:, 1:], axis=0, out=self.prefix[1:, 1:])
        np.cumsum(self.prefix[1:, 1:], axis=1, out=self.prefix[1:, 1:])
        self.diagonal = np.r_[0., np.cumsum(np.diagonal(gram))]
        return self

    def error(self, start, end):
        if end-start < self.min_size:
            raise NotEnoughPoints
        s = self.prefix
        total = s[end,end] - s[start,end] - s[end,start] + s[start,start]
        return self.diagonal[end] - self.diagonal[start] - total/(end-start)


def run_cached(x, params, cache):
    size, jump = int(params['min_size']), int(params['jump'])
    if len(x) < 2*size:
        raise Unsupported(f'PELT-RBF requires at least {2*size} samples with min_size={size}')
    if 'cost' not in cache:
        cache['cost'] = PrefixRbf().fit(x)
    cost = cache['cost']
    penalty = float(params['penalty_multiplier'])*np.log(len(x))
    cuts = rpt.Pelt(custom_cost=cost, min_size=size, jump=jump).fit(x).predict(pen=penalty)[:-1]
    return cuts, dict(source='official ruptures.Pelt + official CostRbf Gram', penalty=penalty,
                      gamma=float(cost.gamma), min_size=size, jump=jump,
                      cost_evaluation='cached prefix sums of unchanged official RBF Gram')
