import argparse
import os
from typing import List

import wandb
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer
from utilities.consts import NEG_VAL, OPTIMIZING_METRIC, SEED_LIST, SINGLE_SEED, NUM_SAMPLES, WANDB_API_KEY, \
    PROJECT_NAME, DATA_PATH, NUM_WORKERS, CPU_PER_TRIAL, GPU_PER_TRIAL
from utilities.utils import reproducible, generate_id


def load_data(conf: argparse.Namespace, is_train: bool = True):
    if is_train:
        train_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            prefetch_factor=5
        )

        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=NUM_WORKERS
        )

        return {'train_loader': train_loader, 'val_loader': val_loader}
    else:

        test_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='test',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=NUM_WORKERS
        )

        return {'test_loader': test_loader}


def start_training(config, checkpoint_dir=None):
    config = argparse.Namespace(**config)
    print(config)

    data_loaders_dict = load_data(config)

    reproducible(config.seed)

    trainer = Trainer(data_loaders_dict['train_loader'], data_loaders_dict['val_loader'], config)

    trainer.run()

    wandb.finish()


def start_testing(config, model_load_path: str):
    config = argparse.Namespace(**config)
    print(config)

    data_loaders_dict = load_data(config, is_train=False)

    reproducible(config.seed)

    tester = Tester(data_loaders_dict['test_loader'], config, model_load_path)

    metric_values = tester.test()
    return metric_values


def start_hyper(conf: dict, model: str, dataset: str, seed: int = SINGLE_SEED):
    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')

    # Search Algorithm
    search_alg = HyperOptSearch(random_state_seed=seed)

    if dataset == 'lfm2b-1mon':
        scheduler = ASHAScheduler(grace_period=4)
    else:
        scheduler = None

    # Logger
    callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key=WANDB_API_KEY,
                                   reinit=True, force=True, job_type='train/val', tags=[model, str(seed), dataset])

    # Hostname
    host_name = os.uname()[1][:2]

    # Dataset
    data_path = DATA_PATH
    conf['data_path'] = os.path.join(data_path, dataset)

    # Seed
    conf['seed'] = seed

    group_name = f'{model}_{dataset}_{host_name}_{seed}'
    tune.register_trainable(group_name, start_training)
    analysis = tune.run(
        group_name,
        config=conf,
        name=generate_id(prefix=group_name),
        resources_per_trial={'gpu': GPU_PER_TRIAL, 'cpu': CPU_PER_TRIAL},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=NUM_SAMPLES,
        callbacks=[callback],
        metric='_metric/' + OPTIMIZING_METRIC,
        mode='max'
    )
    metric_name = '_metric/' + OPTIMIZING_METRIC
    best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
    best_trial_config = best_trial.config
    best_trial_checkpoint = os.path.join(analysis.get_best_checkpoint(best_trial, metric_name, 'max'), 'best_model.pth')

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group='test_results', config=best_trial_config, name=group_name, force=True,
               job_type='test', tags=[model, str(seed), dataset])
    metric_values = start_testing(best_trial_config, best_trial_checkpoint)
    wandb.finish()
    return metric_values


def start_multiple_hyper(conf: dict, model: str, dataset: str, seed_list: List = SEED_LIST):
    print('Starting Multi-Hyperparameter Optimization')
    print('seed_list is ', seed_list)
    metric_values_list = []
    mean_values = dict()

    for seed in seed_list:
        metric_values_list.append(start_hyper(conf, model, dataset, seed))

    for key in metric_values_list[0].keys():
        _sum = 0
        for metric_values in metric_values_list:
            _sum += metric_values[key]
        _mean = _sum / len(metric_values_list)

        mean_values[key] = _mean

    group_name = f'{model}_{dataset}'
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group='aggr_results', name=group_name, force=True, job_type='test',
               tags=[model, dataset])
    wandb.log(mean_values)
    wandb.finish()
