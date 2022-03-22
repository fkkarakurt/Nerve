#include <getopt.h>
#include <stdlib.h>
#include <string.h>

#include "nerveNet.h"
#include "spec.h"

// Hard coded bounds

#define MAX_SIZE 100000
#define MAX_FILENAME_LENGTH 100

// Command line options

char spec_filename[MAX_FILENAME_LENGTH + 1] = "";
char net_filename[MAX_FILENAME_LENGTH + 1] = "";

void parse_options(int argc, char **argv)
{
    int c;

    static const struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
    static const char short_options[] = "h";

    while (1)
    {
        c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        switch (c)
        {
        case 'h':
            printf("Usage:\n");
            printf("  show_network [OPTIONS] specficiation network\n");
            printf("\nDescription:\n");
            printf("  Show the inputs, computed outputs, targets, and error\n");
            printf("  for each of the input/target pairs in the 'specification',\n");
            printf("  using the neural network in 'network'.\n");
            printf("\nOptions:\n");
            printf("  -h, --help\n");
            printf("       display this help and exit\n");
            exit(0);
        case '?':
            printf("Try `show_network --help' for more information.\n");
            exit(1);
        }
    }

    if ((argc != 3))
    {
        printf("show_network: invalid number of arguments\n");
        exit(1);
    }

    strcpy(spec_filename, argv[1]);
    strcpy(net_filename, argv[2]);
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
    float error;
    int i, j;

    parse_options(argc, argv);

    if (read_specification(spec_filename, &no_of_inputs, &no_of_outputs,
                           &no_of_pairs, input, target))
    {
        exit(1);
    }
    net = net_load(net_filename);
    if (net == NULL)
    {
        printf("show_network: failed to load network\n");
        exit(1);
    }
#define inputs(i) (input + i * no_of_inputs)
#define targets(i) (target + i * no_of_outputs)

    for (i = 0; i < no_of_pairs; i++)
    {
        net_compute(net, inputs(i), output);

        error = net_compute_output_error(net, targets(i));

        for (j = 0; j < no_of_inputs; j++)
        {
            printf("%.3f ", *(inputs(i) + j));
        }
        printf("->");
        for (j = 0; j < no_of_outputs; j++)
        {
            printf(" %.3f", *(output + j));
        }
        printf(" (");
        for (j = 0; j < no_of_outputs; j++)
        {
            printf("%.3f", *(targets(i) + j));
            if (j != no_of_outputs - 1)
            {
                printf(" ");
            }
        }
        printf(") %.5f\n", error);
    }

    return 0;
}
