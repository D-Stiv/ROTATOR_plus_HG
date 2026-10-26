# ROTATOR+HG

ROTATOR+HG: Representation Learning for Intermodal Routes with Hypergraph

The code in this repository is an implementation of our published approach: <br>
Steve Gounoue, Genivika Mann, Rajjat Dadwal, and Elena Demidova. “ROTATOR+HG: Representation Learning for Intermodal Routes with Hypergraph”. In: Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26), 2026. DOI: https://doi.org/10.1145/3841645.3843039. <br>
If you use the source code, please cite our research. <br>

<b>Notes:</b> <br>
1) The intermodal route dataset used in this work contains synthetic routes generated over the intermodal graph of the Wangerland region in Germany. The dataset will be made available upon paper publication. <br>
2) Several external data and artifact paths must be configured before running the pretraining and evaluation pipelines. These locations are marked in the source code with `Insert: ...` placeholders. <br>
3) `torch-scatter` must match the installed PyTorch and CUDA versions. If the default installation fails, install the wheel corresponding to your local PyTorch environment. <br>

<b>Introduction:</b> ROTATOR+HG is an intermodal route representation learning approach that jointly captures high-order route dependencies, intermodal characteristics, and context-dependent semantics using hypergraph neural networks. The learned route embeddings can subsequently be evaluated in unsupervised, semi-supervised, and supervised settings. <br>

<b>Implementation Overview:</b> The project follows a multi-stage pipeline consisting of intermodal route generation, spatial-structural embedding pretraining, context-aware route representation learning with ROTATOR+HG, and downstream evaluation. Grid or cell embeddings are pretrained using node2vec, while node and edge embeddings are pretrained using a directed multigraph neural network. These spatial-structural representations, together with node and edge features, are consumed by the ROTATOR+HG pretraining pipeline to produce route-level embeddings. The resulting embeddings can then be evaluated using unsupervised, semi-supervised, and supervised protocols. <br>

<b>Project Directory Structure:</b> The repository is organized into dedicated directories for dataset construction, route generation, ROTATOR+HG pretraining, and downstream evaluation. External datasets, pretrained spatial-structural embeddings, model checkpoints, and other generated artifacts are referenced through paths configured in the corresponding source files. <br>

```text
ROTATOR+HG/
│
├── dataloader/
│   ├── dataset.py
│   └── loader_utils.py
│
├── routes/
│   └── route generation utilities
│
├── pretraining/
│   ├── config.py
│   ├── main.py
│   └── ROTATOR+HG model, losses, and metrics
│
├── evaluation/
│   ├── eval_config.py
│   ├── eval_main.py
│   ├── eval_utils.py
│   ├── finetune.py
│   └── evaluation models and metrics
│
├── requirements.txt
└── LICENSE
```

<b>Folder Descriptions:</b> <br>

<b>dataloader/</b> <br>
Contains the dataset construction and data-loading utilities used by the pretraining and evaluation pipelines. The dataloader combines route instances with the required graph embeddings and node/edge features. Expected inputs include: <br>
+ `train.pkl`, `val.pkl`, and `test.pkl` – route-level training, validation, and test splits <br>
+ node and edge graph embeddings – pretrained graph representations containing `node_id_to_idx`, `edge_id_to_idx`, `node_embeddings`, and `edge_embeddings` <br>
+ cell embeddings – pretrained grid or cell representations <br>
+ node-to-cell mapping – mapping between graph nodes and their corresponding cells <br>
+ edge feature table – structural, scalar, and categorical edge features <br>
+ node feature table – structural, scalar, and categorical node features <br>

Cached dataset objects are written to the dataset-cache directory configured in `dataloader/loader_utils.py`. <br>

<b>routes/</b> <br>
Contains the scripts and utilities required to generate the intermodal routes used by ROTATOR+HG. Route generation relies on the intermodal graph of the Wangerland region and sequences of Points of Interest (POIs). The resulting routes can combine different transportation modes and are subsequently divided into training, validation, and test sets. <br>

<b>pretraining/</b> <br>
Contains the implementation of the ROTATOR+HG representation learning model and its pretraining pipeline. Important components include: <br>
+ `config.py` – pretraining configuration and command-line arguments <br>
+ `main.py` – main pretraining entry point <br>
+ model implementation – ROTATOR+HG architecture for context-aware route representation learning <br>
+ losses – objectives used to learn route representations <br>
+ metrics – metrics used during pretraining <br>

Pretraining checkpoints and the path of the best ROTATOR+HG model are configured in `pretraining/main.py`. <br>

<b>evaluation/</b> <br>
Contains the downstream evaluation pipeline for the learned route embeddings. The implementation supports unsupervised, semi-supervised, and supervised evaluation. Important components include: <br>
+ `eval_config.py` – evaluation configuration and command-line arguments <br>
+ `eval_main.py` – main evaluation entry point <br>
+ `eval_utils.py` – utilities for loading pretrained model arguments and checkpoints <br>
+ `finetune.py` – supervised fine-tuning utilities <br>
+ evaluation models and metrics – downstream prediction heads and evaluation functions <br>

Supervised fine-tuning checkpoints are configured in `evaluation/eval_main.py` and `evaluation/finetune.py`. <br>

<b>Repository Usage:</b> <br>
All required Python packages can be installed by running the following command in the terminal: <br>

```bash
pip install -r requirements.txt
```

`torch-scatter` must be compatible with the installed PyTorch and CUDA versions. If installation through the requirements file fails, install the appropriate `torch-scatter` wheel for your environment. <br>

<b>Dataset and Route Generation:</b> <br>
The experiments use synthetic intermodal routes generated over the intermodal graph $\mathcal{G}$ of the Wangerland region in Germany. The routes represent touristic day trips comprising 1 to 7 Points of Interest (POIs), such as restaurants, parks, museums, and hotels. Their average duration is approximately 1.5 to 2 hours, with a maximum of around 4.5 hours, while travel distances range from a few meters to several kilometers. <br>

The route generation pipeline follows these main steps: <br>
1. Build or obtain the intermodal graph for the Wangerland region. <br>
2. Generate intermodal route instances from the graph and POI sequences. <br>
3. Split the generated routes into training, validation, and test sets. <br>

The resulting route split files `train.pkl`, `val.pkl`, and `test.pkl` are used by the dataloader during ROTATOR+HG pretraining and evaluation. <br>

<b>Spatial-Structural Embedding:</b> <br>
Before context-aware route representation learning, spatial-structural embeddings are pretrained to represent the elements of the underlying intermodal network. <br>

Grid or cell embeddings are pretrained with node2vec following Grover and Leskovec (2016). Node and edge embeddings are pretrained using a directed multigraph neural network following Egressy et al. (2024). These pretrained representations are subsequently loaded by `dataloader/dataset.py` and used as input features by the ROTATOR+HG pretraining pipeline. <br>

In addition to the pretrained embeddings, node and edge feature tables containing structural, scalar, and categorical features must be prepared before running ROTATOR+HG. <br>

<b>ROTATOR+HG Pretraining:</b> <br>
Once the route splits, spatial-structural embeddings, node-to-cell mapping, and node and edge feature tables are available, ROTATOR+HG can be pretrained to learn context-aware route embeddings. Pretraining configuration is defined in `pretraining/config.py`. <br>

Run the following command from the repository root: <br>

```bash
python -m pretraining.main --max_epochs 500 --patience 50 --batch_size 512
```

The generated model checkpoints and the best pretrained ROTATOR+HG model are stored at the locations configured in `pretraining/main.py`. <br>

<b>Route Embedding Evaluation:</b> <br>
Once ROTATOR+HG has been pretrained, the learned route representations can be evaluated using unsupervised, semi-supervised, or supervised protocols. Evaluation configuration is defined in `evaluation/eval_config.py`. The pretrained model arguments and checkpoint are loaded from the locations configured in `evaluation/eval_utils.py`. <br>

For unsupervised and semi-supervised evaluation, run: <br>

```bash
python -m evaluation.eval_main --eval_type un_semi --batch_size 2048
```

For supervised fine-tuning evaluation, run: <br>

```bash
python -m evaluation.eval_main --eval_type finetune-supervised --finetune_epochs 500 --finetune_patience 50 --batch_size 2048
```

The supervised fine-tuning checkpoints are stored at the locations configured in `evaluation/eval_main.py` and `evaluation/finetune.py`. <br>

<b>Complete Pipeline:</b> <br>
The complete ROTATOR+HG workflow should be executed in the following order: <br>
1. Build or obtain the intermodal graph for the Wangerland region. <br>
2. Generate route instances and split them into training, validation, and test sets. <br>
3. Pretrain spatial-structural graph embeddings: node2vec for grid or cell embeddings and directed multigraph GNN embeddings for graph nodes and edges. <br>
4. Prepare node and edge feature tables containing the required structural, scalar, and categorical features. <br>
5. Run ROTATOR+HG pretraining to learn context-aware route embeddings. <br>
6. Evaluate the learned embeddings using unsupervised, semi-supervised, or supervised fine-tuning protocols. <br>

<b>Citation:</b> <br>
If you find this work relevant, please cite us:

```bibtex
@inproceedings{gounoue2026rotatorhg,
  title = {ROTATOR+HG: Representation Learning for Intermodal Routes with Hypergraph},
  author = {Gounoue, Steve and Mann, Genivika and Dadwal, Rajjat and Demidova, Elena},
  year = {2026},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems},
  doi = {10.1145/3841645.3843039}
}
```

<b>Disclaimer: </b> <br>
+ The OpenStreetMap name is a trademark of the OpenStreetMap Foundation and is used with their permission. We are not endorsed by or affiliated with the OpenStreetMap Foundation.
+ This code  is published only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market. <br>


<b>Affiliations: </b> <br> 
The source code is being published in affiliation with the Data Science and Intelligent Systems Group (DSIS), University of Bonn, Germany (https://www.uni-bonn.de/de) and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany (https://lamarr-institute.org/). <br>


<b>License:</b> <br>
This project is released under the license included in `LICENSE`. <br>

<b>Acknowledgements:</b> <br>
This work was partially funded by mFUND, Federal Ministry for Transport (BMV), Germany (MoToRes, 01F2271C), the Deutsche Forschungsgemeinschaft (DFG), Germany (RUSHMORE, 547610252), and the Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany.