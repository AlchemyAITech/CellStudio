from typing import Dict, List

import pandas as pd


class KFoldAggregator:
    """
    Aggregates statistical metrics across K-Folds and generates Mean ± Std formatted strings.
    """
    def __init__(self, metrics_list: List[Dict[str, float]]):
        self.metrics_list = metrics_list

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the raw dataframe containing all folds."""
        return pd.DataFrame(self.metrics_list)

    def summarize(self) -> Dict[str, str]:
        """Returns string representations of Mean ± Std for each metric."""
        df = self.get_dataframe()
        summary = {}
        for col in df.columns:
            # Handle non-numerical or skipped metrics gracefully
            if not pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = "N/A"
                continue
            mean_val = df[col].mean()
            std_val = df[col].std()
            summary[col] = f"{mean_val:.4f} ± {std_val:.4f}"
        return summary
