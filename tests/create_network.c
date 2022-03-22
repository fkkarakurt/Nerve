#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "nerveNet.h"

// Hard coded bounds

#define MAX_LAYERS 10
#define MAX_FILENAME_LENGTH 100

// Command line options

char output_filename[MAX_FILENAME_LENGTH + 1] = "";
int no_of_neurons[MAX_LAYERS];
int no_of_layers;
int use_bias = 1;

void parse_options(int argc, char **argv)
{
    int c;

    static const struct option long_options[] = {
        {"disable-bias", no_argument, NULL, 'd'},
        {"output", required_argument, NULL, 'o'},
        {"specification", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
    static const char short_options[] = "ds:o:h";

    while (1)
    {
        c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        switch (c)
        {
        case 'h':
            printf("Usage:\n");
            printf("  create_network [OPTIONS] m0 ... m(n-1)\n");
#if 0
      printf ("  create_network [OPTIONS] -s specfile m1 ... m(n-2)\n");
#endif
            printf("\nDescription:\n");
            printf("  Create a neural network with n layers; m0 is the number\n");
            printf("  of neurons in the  input layer, m(n-1) is the number of\n");
            printf("  neurons in the output layer, and m1, ..., m(n-2) are the\n");
            printf("  numbers of neurons in the hidden layers.\n");
            printf("\nOptions:\n");
            printf("  -h, --help\n");
            printf("       display this help and exit\n");
            printf("  -d, --disable-bias\n");
            printf("       disable usage of bias neurons\n");
            printf("  -o filename, --output=filename\n");
            printf("       send output to specified file instead of stdout\n");
#if 0
      printf ("  -s filename, --specification=filename\n");
      printf ("       use specification file to determine number of\n");
      printf ("       inputs and outputs\n");
#endif
            exit(0);
        case '?':
            printf("Try `create_network --help' for more information.\n");
            exit(1);
        case 'o':
            if (strlen(optarg) > MAX_FILENAME_LENGTH)
            {
                printf("create_network: filename too long\n");
                exit(1);
            }
            strcpy(output_filename, optarg);
            break;
        case 's':
            printf("create_network: ");
            printf("specification file usage not yet implemented\n");
            exit(1);
        }
    }

    no_of_layers = 0;
    while (optind < argc)
    {
        if (sscanf(argv[optind], "%i", &no_of_neurons[no_of_layers]) == 0)
        {
            printf("create_network: invalid argument\n");
            exit(1);
        }
        no_of_layers++;
        optind++;
        if (no_of_layers > MAX_LAYERS)
        {
            printf("create_network: too many layers\n");
            exit(1);
        }
    }
    if (no_of_layers < 2)
    {
        printf("create_network: not enough layers\n");
        exit(1);
    }
}

// Main

int main(int argc, char **argv)
{
    network_t *net;

    srandom(time(0));

    parse_options(argc, argv);

    net = net_allocate_l(no_of_layers, no_of_neurons);

    if (!use_bias)
    {
        net_use_bias(net, 0);
    }

    if (strlen(output_filename) == 0)
    {
        net_print(net);
    }
    else
    {
        net_save(output_filename, net);
    }

    net_free(net);

    return 0;
}
