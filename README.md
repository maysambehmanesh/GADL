
# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication

PyTorch implementation of the paper:

**[Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication](https://arxiv.org/abs/2509.09597)**  

*[Maysam Behmanesh*](https://maysambehmanesh.github.io/), [Erkan Turan](https://www.lix.polytechnique.fr/~turan/index.html), [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)*

LIX, École Polytechnique, IP Paris

## Overview

Graph alignment, the problem of identifying corresponding nodes across multiple graphs, is fundamental to numerous applications. Most existing unsupervised methods embed node features into latent representations to enable cross-graph comparison without ground-truth correspondences. However, these methods suffer from two critical limitations: the degradation of node distinctiveness due to oversmoothing in GNN-based embeddings, and the misalignment of latent spaces
across graphs caused by structural noise, feature heterogeneity, and training instability, ultimately leading to unreliable node correspondences. We propose a novel framework employing a dual-pass encoder to inject high-frequency discriminability into node features, paired with a geometry-aware functional map module that learns bijective and isometric transformations to align latent spaces while acting as a low-pass filter on correspondences, enforcing smoothness and robustness as a structural prior in map space. Extensive experiments on graph benchmarks demonstrate that our method consistently outperforms existing unsupervised alignment baselines, exhibiting superior robustness to structural inconsistencies and challenging alignment scenarios.

<img width="1023" height="399" alt="image" src="https://github.com/user-attachments/assets/3a5435e3-f506-403f-b5dd-df920d951792" />

## 📁 Structure
```text
├── eigs/                
├── main_GADL.py        
├── model.py            
├── utils.py            
├── load_data.py        
├── algorithm.py       
├── config.yaml         
└── requirements.txt             
```

## ⚙️ Installation
Create a virtual environment (recommended):
```text
python -m venv gadl_env
source gadl_env/bin/activate   # Linux / Mac
# gadl_env\Scripts\activate    # Windows
```
Install dependencies:
```text
pip install -r requirements.txt
```

## 🧪 Configuration

All experiment settings are controlled via:
```text
config.yaml
```
This includes:

- Model hyperparameters
- Training settings
- Dataset paths

## 📊 Data

Download data from [YOUR SOURCE HERE] and place the extracted files in the appropriate directory.

Then update the dataset path in data_path to point to your local data folder.

Precomputed spectral features (eigenvalues/eigenvectors) can be stored in: ``` eigs/ ```

```text
project_root/
├── data
│   ├── douban.mat
├── eigs/
│   ├── Douban Online_Offline
│   │   ├── L1.pth
│   │   ├── L2.pth
│   │   ├── lam1.pth
│   │   ├── lam2.pth
│   │   ├── phi1.pth
│   │   ├── phi2.pth
│   └── ...
```

## 🚀 Usage

Run the main script:
```text
python main_GADL.py --config config.yaml
```


## Citation
If you find this work useful, please cite:
```bash
@article{behmanesh2025gadl,
  title={Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication},
  author={Behmanesh, Maysam and Turan, Erkan and Ovsjanikov, Maks},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  month = {6-11 Jul},
  publisher = {PMLR},
  year={2026}
}
```

## Reference
- Donati et al. [Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence](https://github.com/LIX-shape-analysis/GeomFmaps)
- Attaiki et al. [Understanding and Improving Features Learned in Deep Functional Maps](https://github.com/pvnieo/clover)
- He et al. [T-GAE: Transferable Graph Autoencoder for Network Alignment](https://github.com/Jason-Tree/T-GAE). 

