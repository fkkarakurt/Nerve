#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ocr.h"

float target[10][10] =
    {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

void train_randomly(network_t *net, training_data_t *array, int n_array,
                    int n_trainings)
{
    int i, j;

    for (i = 0; i < n_trainings; i++)
    {
        j = random() % n_array;
        net_compute(net, (float *)array[j].intensity, NULL);
        net_compute_output_error(net, target[array[j].value - '0']);
        net_train(net);
    }
}

void train_batch(network_t *net, training_data_t *array, int n_array)
{
    int i;

    net_begin_batch(net);
    for (i = 0; i < n_array; i++)
    {
        net_compute(net, (float *)array[i].intensity, NULL);
        net_compute_output_error(net, target[array[i].value - '0']);
        net_train_batch(net);
    }
    net_end_batch(net);
}

void train_all(network_t *net, training_data_t *array, int n_array)
{
    int i;

    for (i = 0; i < n_array; i++)
    {
        net_compute(net, (float *)array[i].intensity, NULL);
        net_compute_output_error(net, target[array[i].value - '0']);
        net_train(net);
    }
}

float get_average_error(network_t *net, training_data_t *array, int n_array)
{
    int i;
    float total_error;

    total_error = 0.0;
    for (i = 0; i < n_array; i++)
    {
        net_compute(net, (float *)array[i].intensity, NULL);
        net_compute_output_error(net, target[array[i].value - '0']);
        total_error += net_get_output_error(net);
    }
    return total_error / n_array;
}

void minimize_with_train_all(network_t *net, training_data_t *array, int n_array,
                             training_data_t *validation, int n_validation,
                             int n_batches,
                             float min_error, int max_trainings,
                             int stop_when_error_increases)
{
    float avg_error, best_error;
    int i, n_trainings;
    network_t *best_net;

    printf("\nTraining network...\n\n");
    printf("Every input/target pair from the training set is visited\n");
    printf("and the network is trained using each pair. After %i loop%s\n",
           n_batches, n_batches == 1 ? "" : "s");
    printf("over the training set, the network is computed for\n");
    printf("every input/target pair of the validation set and the\n");
    printf("average error is determined. When this average error drops\n");
    printf("below %f, training is stopped. Also, when %i\n", min_error,
           max_trainings);
    printf("input/target pairs have been trained, training stops.\n");
    if (stop_when_error_increases)
    {
        printf("Training also stops when the averge error on the validation\n");
        printf("set starts to increase. In all cases, the best network\n");
        printf("encountered is written.\n");
    }
    printf("If training doesn't stop, the whole procedure starts again\n");
    printf("with randomly choosing input/target pairs from the training\n");
    printf("set.\n\n");
    printf("The training set consists of %i input/target pairs and\n",
           n_array);
    printf("the validation set consists of %i inputs/target pairs.\n\n",
           n_validation);

    n_trainings = 0;
    best_error = 1000000.0;
    best_net = net_copy(net);
    while (1)
    {
        avg_error = get_average_error(net, validation, n_validation);
        printf("Error: %5.3f  Trainings: %i\n", avg_error, n_trainings);
        if (avg_error < min_error)
        {
            return;
        }
        if (stop_when_error_increases && (avg_error > best_error))
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        if (avg_error < best_error)
        {
            best_error = avg_error;
            net_overwrite(best_net, net);
        }
        if (n_trainings >= max_trainings)
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        for (i = 0; i < n_batches; i++)
        {
            train_all(net, array, n_array);
            n_trainings += n_array;
        }
    }
}

void minimize_with_train_batch(network_t *net, training_data_t *array,
                               int n_array, training_data_t *validation,
                               int n_validation, int n_batches, float min_error,
                               int max_trainings, int stop_when_error_increases)
{
    float avg_error, best_error;
    int i, n_trainings;
    network_t *best_net;

    printf("\nTraining network...\n\n");
    printf("Every input/target pair from the training set is visited\n");
    printf("and the corresponding weight change is computer. At the end\n");
    printf("all weight changes are applied at once. After %i loop%s\n",
           n_batches, n_batches == 1 ? "" : "s");
    printf("over the training set, the network is computed for\n");
    printf("every input/target pair of the validation set and the\n");
    printf("average error is determined. When this average error drops\n");
    printf("below %f, training is stopped. Also, when %i\n", min_error,
           max_trainings);
    printf("input/target pairs have been trained, training stops.\n");
    if (stop_when_error_increases)
    {
        printf("Training also stops when the averge error on the validation\n");
        printf("set starts to increase. In all cases, the best network\n");
        printf("encountered is written.\n");
    }
    printf("If training doesn't stop, the whole procedure starts again\n");
    printf("with randomly choosing input/target pairs from the training\n");
    printf("set.\n\n");
    printf("The training set consists of %i input/target pairs and\n",
           n_array);
    printf("the validation set consists of %i inputs/target pairs.\n\n",
           n_validation);

    n_trainings = 0;
    best_error = 1000000.0;
    best_net = net_copy(net);
    while (1)
    {
        avg_error = get_average_error(net, validation, n_validation);
        printf("Error: %5.3f  Trainings: %i\n", avg_error, n_trainings);
        if (avg_error < min_error)
        {
            return;
        }
        if (stop_when_error_increases && (avg_error > best_error))
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        if (avg_error < best_error)
        {
            best_error = avg_error;
            net_overwrite(best_net, net);
        }
        if (n_trainings >= max_trainings)
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        for (i = 0; i < n_batches; i++)
        {
            train_batch(net, array, n_array);
            n_trainings += n_array;
        }
    }
}

void minimize_with_train_randomly(network_t *net, training_data_t *array,
                                  int n_array, training_data_t *validation,
                                  int n_validation,
                                  int n_trainings_per_loop,
                                  float min_error,
                                  int max_trainings,
                                  int stop_when_error_increases)
{
    float avg_error, best_error;
    int n_trainings;
    network_t *best_net;

    printf("\nTraining network... \n\n");
    printf("A random input/target pair is chosen from the training\n");
    printf("set and the network is trained using this pair. This\n");
    printf("happens %i times. Then the network is computed for\n",
           n_trainings_per_loop);
    printf("every input/target pair of the validation set and the\n");
    printf("average error is computed. When this average error drops\n");
    printf("below %f, training is stopped. Also, when %i\n", min_error,
           max_trainings);
    printf("input/target pairs have been trained, training stops.\n");
    if (stop_when_error_increases)
    {
        printf("Training also stops when the averge error on the validation\n");
        printf("set starts to increase. In all cases, the best network\n");
        printf("encountered is written.\n");
    }
    printf("If training doesn't stop, the whole procedure starts again\n");
    printf("with randomly choosing input/target pairs from the training\n");
    printf("set.\n\n");
    printf("The training set consists of %i input/target pairs and\n",
           n_array);
    printf("the validation set consists of %i inputs/target pairs.\n\n",
           n_validation);

    n_trainings = 0;
    best_error = 1000000.0;
    best_net = net_copy(net);
    while (1)
    {
        avg_error = get_average_error(net, validation, n_validation);
        printf("Error: %5.3f  Trainings: %i\n", avg_error, n_trainings);
        if (avg_error < min_error)
        {
            return;
        }
        if (stop_when_error_increases && (avg_error > best_error))
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        if (avg_error < best_error)
        {
            best_error = avg_error;
            net_overwrite(best_net, net);
        }
        if (n_trainings >= max_trainings)
        {
            net_overwrite(net, best_net);
            net_free(best_net);
            return;
        }
        train_randomly(net, array, n_array, n_trainings_per_loop);
        n_trainings += n_trainings_per_loop;
    }
}
