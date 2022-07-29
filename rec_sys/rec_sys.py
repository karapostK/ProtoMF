from functools import partial

import torch
from torch import nn

from feature_extraction.feature_extractors import FeatureExtractor


class RecSys(nn.Module):

    def __init__(self, n_users: int, n_items: int, rec_sys_param, user_feature_extractor: FeatureExtractor,
                 item_feature_extractor: FeatureExtractor, loss_func_name: str, loss_func_aggr: str = 'mean'):
        """
        General Recommender System
        It generates the user/item vectors (given the feature extractors) and computes the similarity by the dot product.
        :param n_users: number of users in the system
        :param n_items: number of items in the system
        :param rec_sys_param: parameters of the Recommender System module
        :param user_feature_extractor: feature_extractor.FeatureExtractor module that generates user embeddings.
        :param item_feature_extractor: feature_extractor.FeatureExtractor module that generates item embeddings.
        :param loss_func_name: name of the loss function to use for the network.
        :param loss_func_aggr: type of aggregation for the loss function, either 'mean' or 'sum'.
        """

        assert loss_func_aggr in ['mean', 'sum'], f'Loss function aggregators <{loss_func_aggr}> not implemented...yet'

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.rec_sys_param = rec_sys_param
        self.user_feature_extractor = user_feature_extractor
        self.item_feature_extractor = item_feature_extractor
        self.loss_func_name = loss_func_name
        self.loss_func_aggr = loss_func_aggr

        self.use_bias = self.rec_sys_param["use_bias"] > 0 if 'use_bias' in self.rec_sys_param else True

        if self.use_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
            self.item_bias = nn.Embedding(self.n_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        if self.loss_func_name == 'bce':
            self.rec_loss = partial(bce_loss, aggregator=self.loss_func_aggr)
        elif self.loss_func_name == 'bpr':
            self.rec_loss = partial(bpr_loss, aggregator=self.loss_func_aggr)
        elif self.loss_func_name == 'sampled_softmax':
            self.rec_loss = partial(sampled_softmax_loss, aggregator=self.loss_func_aggr)
        else:
            raise ValueError(f'Recommender System Loss function <{self.rec_loss}> Not Implemented... Yet')

        self.initialized = False

        print(f'Built RecSys module \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- user_feature_extractor: {self.user_feature_extractor.name} \n'
              f'- item_feature_extractor: {self.item_feature_extractor.name} \n'
              f'- loss_func_name: {self.loss_func_name} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        """
        Method for initializing the Recommender System Processor
        """
        if self.use_bias:
            torch.nn.init.constant_(self.user_bias.weight, 0.)
            torch.nn.init.constant_(self.item_bias.weight, 0.)

        self.user_feature_extractor.init_parameters()
        self.item_feature_extractor.init_parameters()

        self.initialized = True

    def loss_func(self, logits, labels):
        """
        Loss function of the Recommender System module. It takes into account eventual feature_extractor loss terms.
        NB. Any feature_extractor loss is pre-weighted.
        :param logits: output of the system.
        :param labels: binary labels
        :return: aggregated loss
        """

        rec_loss = self.rec_loss(logits, labels)
        item_feat_ext_loss = self.item_feature_extractor.get_and_reset_loss()
        user_feat_ext_loss = self.user_feature_extractor.get_and_reset_loss()
        return rec_loss + item_feat_ext_loss + user_feat_ext_loss

    def forward(self, u_idxs, i_idxs):
        """
        Performs the forward pass considering user indexes and the item indexes. Negative Sampling is done automatically
        by the dataloader
        :param u_idxs: User indexes. Shape is (batch_size,)
        :param i_idxs: Item indexes. Shape is (batch_size, n_neg + 1)

        :return: A matrix of logits values. Shape is (batch_size, 1 + n_neg). First column is always associated
                to the positive track.
        """
        assert self.initialized, 'Model initialization has not been called! Please call .init_parameters() ' \
                                 'before using the model'

        # --- User pass ---
        u_embed = self.user_feature_extractor(u_idxs)
        if self.use_bias:
            u_bias = self.user_bias(u_idxs)

        # --- Item pass ---
        if self.use_bias:
            i_bias = self.item_bias(i_idxs).squeeze()

        i_embed = self.item_feature_extractor(i_idxs)

        # --- Dot Product ---
        dots = torch.sum(u_embed.unsqueeze(1) * i_embed, dim=-1)  # [batch_size, n_neg_p_1]

        if self.use_bias:
            # Optional bias
            dots = dots + u_bias + i_bias + self.global_bias

        return dots


def bce_loss(logits, labels, aggregator='mean'):
    """
    It computes the binary cross entropy loss with negative sampling, expressed by the formula:
                                    -∑_j log(x_ui) + log(1 - x_uj)
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while
    Item j is a negative instance. The Sum is carried out across the different negative instances. In other words
    the positive item is weighted as many as negative items are considered.

    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to mean

    :return: The binary cross entropy as computed above
    """
    weights = torch.ones_like(logits)
    weights[:, 0] = logits.shape[1] - 1

    loss = nn.BCEWithLogitsLoss(weights.flatten(), reduction=aggregator)(logits.flatten(), labels.flatten())

    return loss


def bpr_loss(logits, labels, aggregator='mean'):
    """
    It computes the Bayesian Personalized Ranking loss (https://arxiv.org/pdf/1205.2618.pdf).

    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to mean

    :return: The bayesian personalized ranking loss
    """
    pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
    neg_logits = logits[:, 1:]  # [batch_size,n_neg]

    labels = labels[:, 0]  # I guess this is just to avoid problems with the device
    labels = torch.repeat_interleave(labels, neg_logits.shape[1])

    diff_logits = pos_logits - neg_logits

    loss = nn.BCEWithLogitsLoss(reduction=aggregator)(diff_logits.flatten(), labels.flatten())

    return loss


def sampled_softmax_loss(logits, labels, aggregator='sum'):
    """
    It computes the (Sampled) Softmax Loss (a.k.a. sampled cross entropy) expressed by the formula:
                        -x_ui +  log( ∑_j e^{x_uj})
    where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while j
    goes over all the sampled items (negatives + the positive).
    :param logits: Logits values from the network. The first column always contain the values of positive instances.
            Shape is (batch_size, 1 + n_neg).
    :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
    :param aggregator: function to use to aggregate the loss terms. Default to sum
    :return:
    """

    pos_logits_sum = - logits[:, 0]
    log_sum_exp_sum = torch.logsumexp(logits, dim=-1)

    sampled_loss = pos_logits_sum + log_sum_exp_sum

    if aggregator == 'sum':
        return sampled_loss.sum()
    elif aggregator == 'mean':
        return sampled_loss.mean()
    else:
        raise ValueError('Loss aggregator not defined')
