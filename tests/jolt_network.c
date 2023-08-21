#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "nerveNet.h"

// Hard coded bounds

#define MAX_FILENAME_LENGTH 100

// Command line options

char filename[MAX_FILENAME_LENGTH + 1] = "";
float range = 0.1;
float factor = 0.1;

void parse_options(int argc, char **argv)
{
    int c;

    static const struct option long_options[] = {
        {"range", required_argument, NULL, 'r'},
        {"factor", required_argument, NULL, 'f'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
    static const char short_options[] = "r:f:h";

    while (1)
    {
        c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        switch (c)
        {
        case 'h':
            printf("Usage:\n");
            printf("  jolt_network [OPTIONS] network\n");
            printf("\nDescription:\n");
            printf("  Make small changes to the weights of a neural network.\n");
            printf("\nOptions:\n");
            printf("  -r, --range=R\n");
            printf("       all weights that are in absolute value smaller than R\n");
            printf("       are replaced by a random weight from the interval\n");
            printf("       [-R,R]. The default value of R is 0.1.\n");
            printf("  -f, --factor=F\n");
            printf("       all other weights are multiplied by a random factor from\n");
            printf("       the interval [1.0-F,1.0+F]. The default value of F is 0.1.\n");
            printf("  -h, --help\n");
            printf("       display this help and exit\n");
            exit(0);
        case 'r':
            sscanf(optarg, "%f", &range);
            break;
        case 'f':
            sscanf(optarg, "%f", &factor);
            break;
        case '?':
            printf("Try `jolt_network --help' for more information.\n");
            exit(1);
        }
    }

    if (argc < 2)
    {
        printf("jolt_network: invalid number of arguments\n");
        printf("Try `jolt_network --help' for more information.\n");
        exit(1);
    }

    strcpy(filename, argv[argc - 1]);
}

// Main

int main(int argc, char **argv)
{
    network_t *net;

    srandom(time(0));

    parse_options(argc, argv);

    net = net_load(filename);
    net_jolt(net, range, factor);
    net_save(filename, net);
    return 0;
}
