# ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations

This repository hosts the code and the additional materials for the paper "ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations" by Alessandro B. Melchiorre, Navid Rekabsaz, Christian Ganhör, and Markus Schedl at RecSys 2022.

**Code** is structured in the following way and briefly described below:

```bash
├── confs
│   └── hyper_params.py
├── data
│   ├── amazon2014
│   │   └── amazon2014_splitter.py
│   ├── lfm2b-1mon
│   │   └── lfm2b-2020_splitter.py
│   └── ml-1m
│       └── movielens_splitter.py
├── experiment_helper.py
├── feature_extraction
│   ├── feature_extractor_factories.py
│   └── feature_extractors.py
├── pdfs and images
│   ├── protomf_appendix.pdf
│   ├── protomf_diagram.png
│   └── ProtoMF__Prototype_based_Matrix_Factorization.pdf
├── protomf.yml
├── README.md
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
