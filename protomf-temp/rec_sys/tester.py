import torch
import wandb
from torch import nn
from torch.utils import data

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSys
from utilities.eval import Evaluator
from utilities.utils import print_results


class Tester:

    def __init__(self, test_loader: data.DataLoader, conf, model_load_path: str):
        """
        Test the model
        :param test_loader: Test DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        :param model_load_path: Path to load the model to test
        """

        self.test_loader = test_loader

        self.rec_sys_param = conf.rec_sys_param
        self.ft_ext_param = conf.ft_ext_param
        self.model_load_path = model_load_path

        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.device = conf.device

        self.model = self._build_model()

        print(f'Built Tester module \n'
              f'- loss_func_name: {self.loss_func_name} \n'
              f'- loss_func_aggr: {self.loss_func_aggr} \n'
              f'- device: {self.device} \n'
              f'- model_load_path: {self.model_load_path} \n')

    def _build_model(self):
        # Step 1 --- Building User and Item Feature Extractors
        n_users = self.test_loader.dataset.n_users
        n_items = self.test_loader.dataset.n_items
        user_feature_extractor, item_feature_extractor = \
            FeatureExtractorFactory.create_models(self.ft_ext_param, n_users, n_items)
        # Step 2 --- Building RecSys Module
        rec_sys = RecSys(n_users, n_items, self.rec_sys_param, user_feature_extractor, item_feature_extractor,
                         self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()

        # Step 3 --- Loading
        params = torch.load(self.model_load_path, map_location=self.device)
        rec_sys.load_state_dict(params)
        rec_sys = rec_sys.to(self.device)
        print('Model Loaded')

        return rec_sys

    @torch.no_grad()
    def test(self):
        """
        Runs the evaluation procedure.

        """
        self.model.eval()
        print('Testing started')
        test_loss = 0
        eval = Evaluator(self.test_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.test_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            test_loss += self.model.loss_func(out, labels).item()

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out)

        test_loss /= len(self.test_loader)

        metrics_values = {**eval.get_results(), 'test_loss': test_loss}

        print_results(metrics_values)

        try:
            wandb.log(metrics_values)
        except wandb.Error:
            print('Not logged to wandb!')

        return metrics_values

    @torch.no_grad()
    def get_test_logits(self):
        """
                Returns the Logits on the Test Dataset
        """
        self.model.eval()
        print('Testing started')

        eval = Evaluator(self.test_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.test_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)

            out = self.model(u_idxs, i_idxs)

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out, sum=False)

        results = eval.get_results(aggregated=False)

        return results
