# Prerequisite
Conda

# Installation

## Install from source
```
git clone https://github.com/inisis/caffe.git
pip install .
```

## Install from pypi
```
pip install brocolli-caffe
```

# How to use
```
PYVER=$(python -c "import sys; print('python{}.{}'.format(*sys.version_info))")
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/$PYVER/site-packages/caffe:$CONDA_PREFIX/lib
```