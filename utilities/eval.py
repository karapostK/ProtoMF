import bottleneck as bn
import numpy as np

from utilities.consts import K_VALUES


def Hit_Ratio_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Hit Ratio. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: HR@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """

    assert logits.shape[1] >= k, 'k value is too high!'

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    hrs = np.any(idx_topk_part[:] == 0, axis=1).astype(int)

    if sum:
        return np.sum(hrs)
    else:
        return hrs


def NDCG_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Normalized Discount Cumulative Gain. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: NDCG@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """
    assert logits.shape[1] >= k, 'k value is too high!'
    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    topk_part = logits[dummy_column, idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[dummy_column, idx_part]

    rows, cols = np.where(idx_topk == 0)
    ndcgs = np.zeros(n)

    if rows.size > 0:
        ndcgs[rows] = 1. / np.log2((cols + 1) + 1)

    if sum:
        return np.sum(ndcgs)
    else:
        return ndcgs


class Evaluator:
    """
    Helper class for the evaluation. When called with eval_batch, it updates the internal results. After the last batch,
    get_results will return the aggregated information for all users.
    """

    def __init__(self, n_users: int, logger=None):
        self.n_users = n_users
        self.logger = logger

        self.metrics_values = {}

    def eval_batch(self, out: np.ndarray, sum: bool = True):
        """
        :param out: Values after last layer. Shape is (batch_size, n_neg + 1).
        """
        for k in K_VALUES:
            for metric_name, metric in zip(['ndcg@{}', 'hit_ratio@{}'], [NDCG_at_k_batch, Hit_Ratio_at_k_batch]):
                if sum:
                    self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k), 0) + \
                                                                 metric(out, k)
                else:
                    self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k),
                                                                                         []) + list(metric(out, k, False))

    def get_results(self, aggregated=True):
        """
        Returns the aggregated results (avg) and logs the results.
        """
        if aggregated:
            for metric_name in self.metrics_values:
                self.metrics_values[metric_name] /= self.n_users

            # Logging if logger is specified
            if self.logger:
                for metric_name in self.metrics_values:
                    self.logger.log_scalar(metric_name, self.metrics_values[metric_name])

        metrics_dict = self.metrics_values
        self.metrics_values = {}

        return metrics_dict
