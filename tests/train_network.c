#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "spec.h"
#include "nerveNet.h"

// Hard coded bounds

#define MAX_FILENAME_LENGTH 100
#define MAX_SIZE 10000

// Command line options

char spec_filename[MAX_FILENAME_LENGTH + 1] = "";
char input_filename[MAX_FILENAME_LENGTH + 1] = "";
char output_filename[MAX_FILENAME_LENGTH + 1] = "";
int max_trainings = 100000000;
int hidden_nodes = 0;
float max_error = 0.0;
int use_bias = 1;

void parse_options(int argc, char **argv)
{
    int c;

    static const struct option long_options[] = {
        {"disable-bias", no_argument, NULL, 'd'},
        {"output", required_argument, NULL, 'o'},
        {"input", required_argument, NULL, 'i'},
        {"error", required_argument, NULL, 'e'},
        {"trainings", required_argument, NULL, 't'},
        {"hidden-nodes", required_argument, NULL, 'n'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
    static const char short_options[] = "dn:o:i:e:t:h";

    while (1)
    {
        c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        switch (c)
        {
        case 'h':
            printf("Usage:\n");
            printf("  train_network [OPTIONS] specification\n");
            printf("\nDescription:\n");
            printf("  Train a neural network towards the function specified\n");
            printf("  in 'specification' by randomly choosing input/target\n");
            printf("  pairs from 'specification'.\n");
            printf("\nOptions:\n");
            printf("  -d, --disable-bias\n");
            printf("       disable usage of bias neurons\n");
            printf("  -e E, --error=E\n");
            printf("       train until the average output error drop below E\n");
            printf("  -t N, --training=T\n");
            printf("       train for at most N trainings\n");
            printf("  -i network, --input=network\n");
            printf("       read a neural network from this file and train\n");
            printf("       this network\n");
            printf("  -o network, --output==network\n");
            printf("       write the trained network to this file; default is\n");
            printf("       stdout\n");
            printf("  -n H --hidden-nodes=H\n");
            printf("       use a neural network with 3 layers and H hidden\n");
            printf("       nodes. Only relevant if -i is not specified.\n");
            printf("  -h, --help\n");
            printf("       display this help and exit\n");
            exit(0);
        case '?':
            printf("Try `train_network --help' for more information.\n");
            exit(1);
        case 'd':
            use_bias = 0;
            break;
        case 't':
            sscanf(optarg, "%i", &max_trainings);
            if (max_trainings == 0)
            {
                max_trainings = 100000000;
            }
            break;
        case 'e':
            sscanf(optarg, "%f", &max_error);
            break;
        case 'o':
            strcpy(output_filename, optarg);
            break;
        case 'n':
            sscanf(optarg, "%i", &hidden_nodes);
            break;
        case 'i':
            strcpy(input_filename, optarg);
            break;
        }
    }

    if (optind >= argc)
    {
        printf("train_network: invalid number of arguments\n");
        exit(1);
    }
    strcpy(spec_filename, argv[optind]);
}

// Main

int main(int argc, char **argv)
{
    network_t *net;
    int no_of_pairs;
    int no_of_inputs;
    int no_of_outputs;
    float input[MAX_SIZE];
    float target[MAX_SIZE];
    float output[MAX_SIZE];
    float error, total_error;
    int t;
    int i;

    srand(time(0));

    parse_options(argc, argv);
    read_specification(spec_filename, &no_of_inputs, &no_of_outputs,
                       &no_of_pairs, input, target);

    if (strlen(input_filename) == 0)
    {
        if (hidden_nodes == 0)
            hidden_nodes = no_of_inputs;
        net = net_allocate(3, no_of_inputs, hidden_nodes, no_of_outputs);
    }
    else
    {
        net = net_load(input_filename);
    }

    if (!use_bias)
    {
        net_use_bias(net, 0);
    }

#define inputs(i) (input + i * no_of_inputs)
#define targets(i) (target + i * no_of_outputs)

    t = 0;
    total_error = 0;
    while ((t < max_trainings) && ((total_error >= max_error) || (t <= 10)))
    {
        i = rand() % no_of_pairs;

        net_compute(net, inputs(i), output);

        error = net_compute_output_error(net, targets(i));

        net_train(net);

        if (t == 0)
        {
            total_error = error;
        }
        else
        {
            total_error = 0.9 * total_error + 0.1 * error;
        }

        t++;
    }

    if (strlen(output_filename) == 0)
    {
        net_print(net);
    }
    else
    {
        net_save(output_filename, net);
    }

    printf("Number of training performed: %i (max %i)\n", t, max_trainings);
    printf("Average output error: %f (max %f)\n", total_error, max_error);

    return 0;
}
