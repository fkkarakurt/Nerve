![Social](https://raw.githubusercontent.com/fkkarakurt/Nerve/main/social.png)


# Nerve | Neural Network Library

This is a basic implementation of a neural network for use in C and C++ programs. It is intended for use in applications that just happen to need a simple neural network and do not want to use needlessly complex neural network libraries.

It features multilayer backpropagation neural network with settable momentum and learning rate, easy portability, and small size.


## Features

- Multilayer perceptron neural network.
- Backpropagation training.
- Trainable bias.
- Small.
- Fast.
- Easy to incorporate in own application.
- Easy to extend.
- Licenced under MIT.
- Includes example application to train a network to recognize handwritten digits.
  
## Internal Structure

![internatl structure](https://raw.githubusercontent.com/fkkarakurt/Nerve/main/structure.png)

## Usage

> This project has been tested with Debian and Fedora.

#### Computing a neural network

For example, to create a neural network with 2 neurons in the input layer, 3 neurons in the hidden layer, and 1 neuron in the output layer, you can use a command like this:

```bash
make
./tests/show_network 3 2 1
```

## Special Thanks

- [Ethem Alpaydın](https://www.cmpe.boun.edu.tr/~ethem/) For datasets and training set & books.
- [Cenk Kaynak](https://www.linkedin.com/in/cenk-kaynak-phd-631aa4101/?originalSubdomain=uk) For datasets and training set & articles.
- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) For datasets.
- [FANN](http://leenissen.dk/fann/wp/)
- [Machine Learning, Tom Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
- [comp.ai.neural-nets FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/)
- [Peter Van Rossum](https://dblp.org/pid/98/3298.html)


## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

  
