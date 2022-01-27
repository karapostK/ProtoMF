# ProtoMF: Prototype-based Matrix Factorization for Explainable Recommendations (temporary anonymized repository)

This temporary repository hosts the code and additional materials for the submission "ProtoMF: Prototype-based Matrix Factorization for ExplainableRecommendations" at SIGIR 2022.


**Evaluation results using other thresholds k (@5,@50)** are located at ```imgs/evaluation_results_using_other_thresholds_k.png```.

**A T-SNE visualization of the items and the prototypes on LFM2b-1mon** is located at ```imgs/Item_proto_1_latent_space.pdf```

**Prototypes from LFM2b-1mon** are shown in ```imgs/prototypes_from_lfm2b-1m.png``` and briefly described below.

The three user prototypes present different music preferences. Prototype 16's top tracks are from Rock/Hard Rock bands, while prototype 35's recommendations belong all to female pop-singers. Prototype 30, instead, prefers Electronic and Downtempo music. Similarly, the three item prototypes capture different music genres. In fact, prototype 6's top neighbors are Lo-fi tracks, while prototype 16's are mostly Hip Hop and Rap. Lastly, prototype 13 represents a prototypical Heavy-Metal track.

**Code** is structured in the following way and briefly described below:

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
