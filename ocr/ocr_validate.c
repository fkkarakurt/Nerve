#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nerveNet.h"
#include "ocr.h"

// Hard coded bounds

#define MAX_FILENAME_LENGTH 100
#define MAX_SET_SIZE 20000

// Command line options

char network_filename[MAX_FILENAME_LENGTH + 1] = "";
char validation_filename[MAX_FILENAME_LENGTH + 1] = "";

void print_usage()
{
    printf("Usage:\n");
    printf("  ocr_validate [OPTIONS] network validation_data\n");
    printf("\nDescription:\n");
    printf("  Try to recognize every character in 'validation_data'\n");
    printf("  using the neural network in 'network'. For every\n");
    printf("  character, the output consists of: the actual character,\n");
    printf("  outputs of the neural network corresponding to the\n");
    printf("  digits 0,1,...,9, recognized character, ok/wrong.\n");
    printf("\nOptions:\n");
    printf("  -h, --help\n");
    printf("       display this help and exit\n");
}

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
            print_usage();
            exit(0);
        }
    }

    if (argc != optind + 2)
    {
        printf("ocr_validate: invalid number of arguments\n");
        printf("Try 'ocr_validate -h' for more information.\n");
        exit(1);
    }

    strcpy(network_filename, argv[1]);
    strcpy(validation_filename, argv[2]);
}

// Main

network_t *net;
training_data_t validation[MAX_SET_SIZE];
int no_validation;

int main(int argc, char **argv)
{
    float output[10], max;
    int i, j, maxj, total, errors;

    parse_options(argc, argv);

    printf("Reading network from '%s'... ", network_filename);
    fflush(stdout);
    net = net_load(network_filename);
    if (net == NULL)
    {
        printf("failed.\n");
        printf("ocr_train: can't open network file '%s' for reading\n",
               network_filename);
        exit(1);
    }
    printf("done.\n");

    printf("Reading validation data from '%s'... ", validation_filename);
    fflush(stdout);
    no_validation = read_training_file(validation_filename, validation);
    printf("done (%i).\n", no_validation);

    total = 0;
    errors = 0;
    for (i = 0; i < no_validation; i++)
    {
        net_compute(net, (float *)validation[i].intensity, output);

        maxj = 0;
        max = -1.0;
        for (j = 0; j < 10; j++)
        {
            if (max < output[j])
            {
                max = output[j];
                maxj = j;
            }
        }

        printf("%c -> ", validation[i].value);
        for (j = 0; j < 10; j++)
            printf("%.3f ", output[j]);
        printf("-> %i -> ", maxj);
        if (maxj == validation[i].value - '0')
        {
            printf("ok\n");
        }
        else
        {
            printf("wrong\n");
            errors++;
        }
        total++;
    }

    printf("Errors: %i out of %i (%.1f%%).\n", errors, total,
           100.0 * errors / total);

    net_free(net);
    return 0;
}
