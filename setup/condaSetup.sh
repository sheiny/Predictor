conda create --name Tensorflow
conda install -c conda-forge cudatoolkit=11.8.0 -y
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

conda install pandas -y
conda install scikit-learn -y
conda install chardet -y
conda install matplotlib -y

conda install ipykernel -y
python -m ipykernel install --user --name=Tensorflow

