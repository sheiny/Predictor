python -m venv Tensorflow
source Tensorflow/bin/activate
pip install --upgrade pip
pip install -r Predictor/requirements.txt
python -m ipykernel install --user --name=Tensorflow
