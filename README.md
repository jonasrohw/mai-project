# relevant-evidence-detection

Original paper: https://doi.org/10.48550/arXiv.2311.09939
Orignal Implementation: https://github.com/stevejpapad/relevant-evidence-detection


## Reproduce

- Clone this repo: 
```
git clone https://github.com/jonasrohw/mai-project
cd relevant-evidence-detection
python src/main.py
```

- Create a python (>= 3.9) environment (Anaconda is recommended) 
- Install all dependencies with: `pip install -r requirements.txt`.

## Datasets

If you want to reproduce the experiments on the paper it is necessary to first download the following datasets and save them in their respective folder: 
- VisualNews -> https://github.com/FuxiaoLiu/VisualNews-Repository -> `data/VisualNews/`
- NewsCLIPings -> https://github.com/g-luo/news_clippings -> `data/news_clippings/`
- NewsCLIPings evidence -> https://github.com/S-Abdelnabi/OoC-multi-modal-fc -> `data/news_clippings/queries_dataset/`
- VERITE -> https://github.com/stevejpapad/image-text-verification -> `data/VERITE/` (Although we provide all necessary files as well as the external evidence) 

Folders:
```
├── README.md
├── checkpoints_pt
├── create-subset.py
├── data
│   ├── VERITE
│   ├── VisualNews
│   │   └── origin
│   └── news_clippings
│       └── queries_dataset
├── requirements.txt
├── results
├── setup_mnt_data.sh
├── src
│   ├── experiment.py
│   ├── extract_features.py
│   ├── main.py
│   ├── models.py
│   ├── models_dynamic.py
│   ├── prepare_VERITE.py
│   ├── prepare_evidence.py
│   ├── utils.py
│   └── utils_evidence.py
└── verite-healing.py
```