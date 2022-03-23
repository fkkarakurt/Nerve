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

#### Computing a neural network

This code snippet shows how to compute the outputs of a neural network with `nerveNet`.

```c
#include "nerveNet.h"

float *input;
float *output;
network_t *net;

net = net_load(filename of neural network);

// Set input to an array of floats that forms the input of the network
// Set output to where you want the ouput of the network stored

net_compute(net, input, output);

// Use the network outputs in output[0], ...  ouput[net_no_of_ouputs(net)-1]
```

#### Training a neural network

The following code snippet shows how you typically train a neural network with `nerveNet`. It changes all the weights once for each input/target pair in a training set. You would typically repeat the loop quite a few times until you're satisfied with the performance of the network.

```c
#include "nerveNet.h"

float *input;
float *target;
network_t *net;
int i;

// Read or create the neural network

for (i = 0; i < number of elements in training set; i++)
{
  // Set input to i-th input element of training set
  // Set target to intended output for this input
  net_compute(net, input, NULL);
  net_compute_output_error(net, target);
  net_train(net);
}
```

#### Training a neural network in batches

The following code snippet does the same; the only difference is that the weights only get changed after all input/target pairs from the training set have been considered.

```c
#include "nerveNet.h"

float *input;
float *target;
network_t *net;
int i;

// Read or create the neural network

net_begin_batch(net);
for (i = 0; i < number of elements in training set; i++)
{
  // Set input to i-th input element of training set;
  // Set target to intended output for this input;
  net_compute(net, input, NULL);
  net_compute_output_error(net, target);
  net_train_batch(net);
}
net_end_batch(net);
```

#### Training a neural network and using a validation set to stop training

The previous two code snippets show how to train a neural network, but it is not yet clear when to stop training. This code snippets gives one way to determine that. Training will stop when the average error on the validation set drops below a certain value or when training has been going on for a long while.

```c
#include "nerveNet.h"

float *input;
float *target;
float error;
network_t *net;
int i, j, count;

// Read or create the neural network

count = 0;
do
{
  // Train the network with all pairs in the training set
  for (i = 0; i < number of elements in training set; i++)
  {
    // Set input to i-th input element of training set
    // Set target to intended output for this input
    net_compute(net, input, NULL);
    net_compute_output_error(net, target);
    net_train(net);
    count++;
  }

  // Compute the average error for all pairs in the validation set
  error = 0.0;
  for (j = 0; j < number of elements in validation set; j++)
  {
    // Set input to i-th input element of validation set
    // Set target to intended output for this input
    net_compute(net, input, NULL);
    net_compute_output_error(net, target);
    error += net_get_output_error(net);
  }
  error /= number of elements in validation set;

  /* Keep going until either the average error is small enough
   * or a large amount of trainings have been completed. */
} while ((error > maximal error that is still acceptable) &&
         (count < maximal number of trainings you want to do));

```
## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

  
