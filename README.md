# Perovskite-Ordering-Descriptors
Repo for our paper "[Data-Driven Physics-Informed Descriptors of Cation Ordering in Multicomponent Perovskite Oxides](https://doi.org/10.1016/j.xcrp.2024.101942)".

## Usage
This repository contains all the data and codes to reproduce the figures of this work.

The Jupyter notebooks require the following packages to run correctly:
```
scipy          1.8.1
numpy          1.22.3
pandas         1.5.3
seaborn        0.12.2
matplotlib     3.7.1
mscorefonts    0.0.1
scikit-learn   1.8.1
```

All these packages can be installed using the [environment.yml](environment.yml) file and `conda`:
```
conda env create -f environment.yml
conda activate pvsk_order_des
```

## Citation
If you use this code, please cite the following paper:
```
@article{peng2024data,
  title={Data-Driven Physics-Informed Descriptors of Cation Ordering in Multicomponent Perovskite Oxides},
  author={Jiayu Peng and James Damewood and Rafael Gómez-Bombarelli},
  journal={Cell Reports Physical Science},
  url = {https://doi.org/10.1016/j.xcrp.2024.101942},
  year={2024}
}
```
