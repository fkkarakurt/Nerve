#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nerveNet.h"
#include "ocr.h"

// Hard coded bounds

#define MAX_FILENAME_LENGTH 100
#define MAX_SET_SIZE 20000

// Command line options

char network_filename[MAX_FILENAME_LENGTH + 1] = "";
char training_filename[MAX_FILENAME_LENGTH + 1] = "";
char validation_filename[MAX_FILENAME_LENGTH + 1] = "";

int batch = 0;
int create_network = 0;
int hidden_nodes = 10;
float max_error = 0.01;
int max_trainings = 10000000;
int stop_err_inc = 0;

void print_usage()
{
    printf("Usage:\n");
    printf("  ocr_train [OPTIONS] network training_data validation_data\n");
    printf("\nOptions:\n");
    printf("  -b B, --batch=B\n");
    printf("       Train the network in batches. Each batch consists of\n");
    printf("       of a loop over the whole training set. After every B\n");
    printf("       batches, network is checked against the validation set.\n");
    printf("  -c, --create-network\n");
    printf("       Create a new network instead of reading one from a\n");
    printf("       file.\n");
    printf("  -n H, --hidden-nodes=H\n");
    printf("       When creating a new network give it H hidden nodes.\n");
    printf("       Default for H is 10.\n");
    printf("  -e E, --error=E\n");
    printf("       Stop training when the average error on the validation\n");
    printf("       set is less than E.\n");
    printf("  -t T, --trainings=T\n");
    printf("	  Stop training when T input/target pairs have been\n");
    printf("       trained.\n");
    printf("  -f, --first-bump\n");
    printf("       Stop training when the error on the validation set\n");
    printf("       starts to increase.\n");
    printf("  -h, --help\n");
    printf("       Display this help and exit.\n");
}

void parse_options(int argc, char **argv)
{
    int c;

    static const struct option long_options[] = {
        {"batch", required_argument, NULL, 'b'},
        {"create-network", no_argument, NULL, 'c'},
        {"first-bump", no_argument, NULL, 'f'},
        {"hidden-nodes", required_argument, NULL, 'n'},
        {"error", required_argument, NULL, 'e'},
        {"trainings", required_argument, NULL, 't'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0}};
    static const char short_options[] = "b:cfn:e:t:h";

    while (1)
    {
        c = getopt_long(argc, argv, short_options, long_options, NULL);
        if (c == -1)
            break;
        switch (c)
        {
        case 'b':
            sscanf(optarg, "%i", &batch);
            break;
        case 'c':
            create_network = 1;
            break;
        case 'f':
            stop_err_inc = 1;
            break;
        case 'n':
            sscanf(optarg, "%i", &hidden_nodes);
            break;
        case 'e':
            sscanf(optarg, "%f", &max_error);
            break;
        case 'h':
            print_usage();
            exit(0);
        }
    }

    if (argc != optind + 3)
    {
        printf("ocr_train: invalid number of arguments\n");
        printf("Try 'ocr_train -h' for more information.\n");
        exit(1);
    }

    strcpy(network_filename, argv[optind]);
    strcpy(training_filename, argv[optind + 1]);
    strcpy(validation_filename, argv[optind + 2]);
}

// Main

training_data_t training[MAX_SET_SIZE];
training_data_t validation[MAX_SET_SIZE];
int no_training, no_validation;
network_t *net;

int main(int argc, char **argv)
{
    srandom(time(0));

    parse_options(argc, argv);

    if (create_network)
    {
        printf("Creating network with %i hidden nodes... ", hidden_nodes);
        fflush(stdout);
        net = net_allocate(3, 64, hidden_nodes, 10);
    }
    else
    {
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
    }
    printf("done.\n");

    printf("Reading training data from '%s'... ", training_filename);
    fflush(stdout);
    no_training = read_training_file(training_filename, training);
    printf("done (%i).\n", no_training);

    if (validation_filename[0] != 0)
    {
        printf("Reading validation data from '%s'... ", validation_filename);
        fflush(stdout);
        no_validation = read_training_file(validation_filename, validation);
        printf("done (%i).\n", no_validation);
    }

    if (batch > 0)
    {
        minimize_with_train_batch(net, training, no_training, validation,
                                  no_validation, batch, max_error,
                                  max_trainings, stop_err_inc);
    }
    else if (batch == 0)
    {
        minimize_with_train_randomly(net, training, no_training, validation,
                                     no_validation, no_training, max_error,
                                     max_trainings, stop_err_inc);
    }
    else
    {
        minimize_with_train_all(net, training, no_training, validation,
                                no_validation, -batch, max_error,
                                max_trainings, stop_err_inc);
    }

    printf("Average error: %f\n",
           get_average_error(net, validation, no_validation));

    net_save(network_filename, net);

    net_free(net);

    return 0;
}
