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

### Create Network

For example, to create a neural network with 2 neurons in the input layer, 3 neurons in the hidden layer, and 1 neuron in the output layer, you can use a command like this:

```bash
make
./create_network 3 2 1
```

> You can modify the following `*.spec` and `*.net` files according to your own testing.

### Training

```bash
./train_network xornet.spec 
```

### Show Network

```bash
./show_network xornet.spec xornet.net 
```

> You can use the `-h` or `--help` arguments for all other methods. This will give you information on usage.

### XOR Example

`test.c`:

```c
#include "nerveNet.h"
#include <stdio.h>
#include <math.h>

int main() {
    // Build the neural network
    
    // 2 input neurons, 2 hidden neurons and 1 output neuron
    int layers[] = {6, 5, 4}; 
    
    // a 3-layer network
    network_t* net = net_allocate_l(3, layers); 

    // Training data
    
    // of type float
    float inputs[][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    // of type float
    float outputs[][1] = { 
        {0},
        {1},
        {1},
        {0}
    };

    // Train network

    // training for 10,000 iterations
    for (int i = 0; i < 10000; i++) { 
        for (int j = 0; j < 4; j++) {
            net_compute(net, inputs[j], NULL); 
        }
    }

    // Test
    float result[1];
    for (int i = 0; i < 4; i++) {
        net_compute(net, inputs[i], result);
        printf("Input: %f, %f -> Output: %f\n", inputs[i][0], inputs[i][1], result[0]);
    }

    // Free the net
    net_free(net);

    return 0;
}

```

Compile:

`gcc test.c network.c -o Test`

Run:

`./Test`

The results still need to be improved. Because in XOR tests:
- 0 ⊕ 0 = 0
- 0 ⊕ 1 = 1
- 1 ⊕ 0 = 1
- 1 ⊕ 1 = 0
we await the results. However, Nerve can give results sometimes close and sometimes far. Try this by changing the number of input, hidden, and output neurons. You can also change the number of iterations. Different results mean not fully trained. But I hope to at least get closer to the correct results.


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

  
