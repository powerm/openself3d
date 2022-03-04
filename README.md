# self-supervised object dense descriptor with 3d data and color data

the algorithm lib  designer using mmcv and borrow some code from mmdet repo.


# Requirements
- Ubuntu 18.04
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch  v1.10.1 or higher
- MinkowskiEngine v0.5 or higher
- mmcv

# Installation & Dataset Download
We recommand conda for installation. First, create a conda environment with pytorch 1.10 or higher with

```python 
conda create -n openself3d python=3.7
conda activate openself3d
conda install pytorch -c pytorch
conda install openblas-devel openblas
# use local MinkowskiEngine
git clone -b  v0.5.1   https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
Next, download openself3d git repository and install the requirement from the openself3d root directory..

```python
git clone https://github.com/powerm/openself3d.git
cd openself3d
# Do the following inside the conda environment
pip install -r requirements.txt
```
