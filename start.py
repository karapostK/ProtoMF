import argparse
import os

from confs.hyper_params import mf_hyper_params, anchor_hyper_params, user_acf_hyper_params, \
    item_acf_hyper_params, user_proto_standard_hyper_params, user_proto_shifted_hyper_params, \
    user_proto_shifted_and_div_hyper_params, user_proto_chose_hyper_params, item_proto_chose_hyper_params, \
    proto_double_tie_chose_hyper_params, user_proto_chose_original_hyper_params, item_proto_chose_original_hyper_params, \
    proto_double_tie_chose_original_hyper_params, user_proto_chose_0_reg_hyper_params, \
    item_proto_chose_0_reg_hyper_params, proto_double_tie_chose_0_reg_hyper_params, \
    proto_double_tie_chose_shifted_and_div_hyper_params
from experiment_helper import start_hyper, start_multiple_hyper
from utilities.consts import SINGLE_SEED

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--model', '-m', type=str, help='Recommender System Model',
                    choices=['mf', 'acf', 'user_proto', 'item_proto', 'user_item_proto', 'user_acf', 'item_acf',
                             'user_proto_standard', 'user_proto_shifted', 'user_proto_shifted_and_div',
                             'user_proto_chose', 'item_proto_chose', 'proto_double_tie_chose',
                             'user_proto_chose_original', 'item_proto_chose_original',
                             'proto_double_tie_chose_original', 'user_proto_chose_0_reg', 'item_proto_chose_0_reg',
                             'proto_double_tie_chose_0_reg', 'proto_double_tie_chose_shifted_and_div'])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'lfm2b-1y'])

parser.add_argument('--multiple', '-mp', action='store_true', default=False, required=False)
parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED, required=False)

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
elif model == 'user_acf':
    conf_dict = user_acf_hyper_params
elif model == 'item_acf':
    conf_dict = item_acf_hyper_params
elif model == 'user_proto_standard':
    conf_dict = user_proto_standard_hyper_params
elif model == 'user_proto_shifted':
    conf_dict = user_proto_shifted_hyper_params
elif model == 'user_proto_shifted_and_div':
    conf_dict = user_proto_shifted_and_div_hyper_params
elif model == 'user_proto_chose':
    conf_dict = user_proto_chose_hyper_params
elif model == 'item_proto_chose':
    conf_dict = item_proto_chose_hyper_params
elif model == 'proto_double_tie_chose':
    conf_dict = proto_double_tie_chose_hyper_params
elif model == 'user_proto_chose_original':
    conf_dict = user_proto_chose_original_hyper_params
elif model == 'item_proto_chose_original':
    conf_dict = item_proto_chose_original_hyper_params
elif model == 'proto_double_tie_chose_original':
    conf_dict = proto_double_tie_chose_original_hyper_params
elif model == 'user_proto_chose_0_reg':
    conf_dict = user_proto_chose_0_reg_hyper_params
elif model == 'item_proto_chose_0_reg':
    conf_dict = item_proto_chose_0_reg_hyper_params
elif model == 'proto_double_tie_chose_0_reg':
    conf_dict = proto_double_tie_chose_0_reg_hyper_params
elif model == 'proto_double_tie_chose_shifted_and_div':
    conf_dict = proto_double_tie_chose_shifted_and_div_hyper_params

if multiple:
    start_multiple_hyper(conf_dict, model, dataset)
else:
    start_hyper(conf_dict, model, dataset, seed)
