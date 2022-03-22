#include <stdio.h>

#define MAX_LINE_LENGTH 1000

int read_specification(char *filename, int *no_of_inputs, int *no_of_outputs,
                       int *no_of_pairs, float *input, float *target)
{
    FILE *file;
    char line[MAX_LINE_LENGTH + 1], *index;
    float *in, *tar;
    int skipped, i;

    file = fopen(filename, "r");

    if (file == NULL)
    {
        fprintf(stderr, "read_specification: can't open %s for reading\n",
                filename);
        return -1;
    }

    do
    {
        fgets(line, MAX_LINE_LENGTH, file);
    } while (line[0] == '#');
    sscanf(line, "%i\n", no_of_inputs);

    do
    {
        fgets(line, MAX_LINE_LENGTH, file);
    } while (line[0] == '#');
    sscanf(line, "%i\n", no_of_outputs);

    *no_of_pairs = 0;
    in = input;
    tar = target;
    fgets(line, MAX_LINE_LENGTH, file);
    do
    {
        if (line[0] != '#')
        {
            index = line;
            for (i = 0; i < *no_of_inputs; i++)
            {
                sscanf(index, "%f%n", in++, &skipped);
                index += skipped;
            }
            fgets(line, MAX_LINE_LENGTH, file);
            index = line;
            for (i = 0; i < *no_of_outputs; i++)
            {
                sscanf(index, "%f%n", tar++, &skipped);
                index += skipped;
            }
            (*no_of_pairs)++;
        }
        fgets(line, MAX_LINE_LENGTH, file);
    } while (!feof(file));
    return 0;
}
