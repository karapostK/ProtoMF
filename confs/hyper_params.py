import torch
from ray import tune

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 100,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
}

base_hyper_params = {
    **base_param,
    'neg_train': tune.randint(1, 50),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'batch_size': tune.lograndint(64, 512, 2),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1)
    },
}

mf_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "detached",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "embedding",
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}
anchor_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'sum',
    'ft_ext_param': {
        "ft_type": "acf",
        'embedding_dim': tune.randint(10, 100),
        'n_anchors': tune.randint(10, 100),
        'delta_exc': tune.loguniform(1e-2, 10),
        'delta_inc': tune.loguniform(1e-2, 10),
    },
}

user_proto_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "prototypes",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}

item_proto_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            "ft_type": "prototypes",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
        'user_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}
proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
        'user_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
    },
}
