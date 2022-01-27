# ProtoMF: Prototype-based Matrix Factorization for ExplainableRecommendations (temporary anonymized repository)

This temporary repository hosts the code for the submission "ProtoMF: Prototype-based Matrix Factorization for ExplainableRecommendations" at SIGIR 2022.

![](https://github.com/karapostK/ProtoMF-temp/blob/04061e11b809a702a63ef898370027e40519573e/protomf_diagram.png "sutff")

Code is structured in the following way and briefly described below:

```bash
├── confs
│   └── hyper_params.py
├── data
│   ├── amazon2014
│   │   └── amazon2014_splitter.py
│   ├── lfm2b-1y
│   │   └── lfm2b-1y_splitter.py
│   └── ml-1m
│       └── movielens_splitter.py
├── experiment_helper.py
├── feature_extraction
│   ├── feature_extractor_factories.py
│   └── feature_extractors.py
├── protorec.yml
├── rec_sys
│   ├── protorec_dataset.py
│   ├── rec_sys.py
│   ├── tester.py
│   └── trainer.py
├── start.py
└── utilities
    ├── consts.py
    ├── eval.py
    └── utils.py

```
where the files/directories contain:
- `protorec.yml`: environment (install with `conda env create -f protorec.yml`)
- `start.py`: starting point to run the experiments (check with `python start.py --help`)
- `experiment_helper.py`: hyperparameter optimization
- `confs/hyper_params.py`: hyperparameters of all the models
- `data/*`: data splitting procedure described for each dataset
- `feature_extraction/*`: code of the models 
- `rec_sys/protorec_dataset.py`: general code for handling the dataset (including negative sampling)
- `rec_sys/rec_sys.py`: a matrix-factorization-based recommender system, used as base for all models
- `rec_sys/tester.py` and `rec_sys/trainer.py`: testing and training procedure respectively
- `utilities/*`: constants, evaluation metrics code, generic code
