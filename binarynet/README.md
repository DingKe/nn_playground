An implemtation of binaryNet for Keras.

The binarized Dense and Conv2D are two keras layers, thus can be integrated into keras framework out of box.

## To run the demo:
### train a binary MLP model on MNIST
python mnist_mlp.py 
### train a binary CNN model on MNIST
python mnist_cnn.py 

The code is according to the [theano version](https://github.com/MatthieuCourbariaux/BinaryNet).
The only missing ingredient is that the learning rate is not scaled w.r.t. weight' fan-in & fan-out. 
(An involved [patch](https://github.com/fchollet/keras/pull/3004) is needed.)

## Reference
* Courbariaux et al. [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830).
