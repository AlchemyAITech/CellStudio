import numpy as np
import pandas as pd
from ..registry import MetricRegistry
from ..base import BaseMetric
@MetricRegistry.register("ICC")
class ICC(BaseMetric):
    def __init__(self, **kwargs): pass
    def compute(self, y_true, y_pred, y_prob=None, **kwargs) -> float:
        try:
            import pingouin as pg
            n = len(y_true)
            df = pd.DataFrame({'Target': np.concatenate([np.arange(n), np.arange(n)]), 'Rater': np.concatenate([np.zeros(n), np.ones(n)]), 'Score': np.concatenate([y_true, y_pred])})
            icc_res = pg.intraclass_corr(data=df, targets='Target', raters='Rater', ratings='Score')
            return float(icc_res.set_index('Type').loc['ICC2', 'ICC'])
        except Exception:
            return 0.0
