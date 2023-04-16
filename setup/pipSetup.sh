python -m venv Tensorflow
source Tensorflow/bin/activate
pip install --upgrade pip
pip install -r ./pipRequirements.txt
python -m ipykernel install --user --name=Tensorflow
