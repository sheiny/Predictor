# Predictor
A DRV predictor using TensorFlow library.

# How to Install:
Please Install these packages in you env:<br/>
0- conda install nb_conda<br/>
1- conda install matplotlib<br/>
2- conda install tensorflow<br/>
3- conda install scikit-learn<br/>
4- conda install networkx<br/>
5- conda install seaborn<br/>

# How to run:
How to run it using jupyter-notebook:<br/>
Please fix all file paths, and run these cells in this order:<br/>

1- Imports<br/>
2- Hyperparameters<br/>
3- Data preprocessing and graph functions using NetworkX library<br/>
4- Some plot functions<br/>
5- Load Training Data from JSON Grid Graphs<br/>
6- Convolutional Neural Network<br/>
7- Train with Image Generators<br/>
8- Test with Image Generators<br/>
9- Training performance<br/>
10- Test performance<br/>

Observations: when training it should be saving the model after each epoch (almost 1 hour), so if you want, you can re-start the process from where you left.
The history and the model, are being saved temporarily  in ./checkpoints/
