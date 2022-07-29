import argparse
import os

from confs.hyper_params import mf_hyper_params, anchor_hyper_params, user_proto_chose_original_hyper_params, \
    item_proto_chose_original_hyper_params, proto_double_tie_chose_original_hyper_params
from experiment_helper import start_hyper, start_multiple_hyper
from utilities.consts import SINGLE_SEED

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--model', '-m', type=str, help='Recommender System model',
                    choices=['mf', 'acf', 'user_proto', 'item_proto', 'user_item_proto'])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'lfm2b-1mon'])

parser.add_argument('--multiple', '-mp',
                    help='Whether to run the experiment across all seeds (see utilities/consts.py)',
                    action='store_true', default=False, required=False)
parser.add_argument('--seed', '-s', help='Seed to set for the experiments', type=int, default=SINGLE_SEED,
                    required=False)

args = parser.parse_args()

model = args.model
dataset = args.dataset
multiple = args.multiple
seed = args.seed

conf_dict = None
if model == 'mf':
    conf_dict = mf_hyper_params
elif model == 'acf':
    conf_dict = anchor_hyper_params
elif model == 'user_proto':
    conf_dict = user_proto_chose_original_hyper_params
elif model == 'item_proto':
    conf_dict = item_proto_chose_original_hyper_params
elif model == 'user_item_proto':
    conf_dict = proto_double_tie_chose_original_hyper_params

if multiple:
    start_multiple_hyper(conf_dict, model, dataset)
else:
    start_hyper(conf_dict, model, dataset, seed)
