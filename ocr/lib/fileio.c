#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include "ocr.h"

void fscan_training_digit(gzFile file, training_data_t *data)
{
    int x, y;
    char digit[32][32];
    char line[100];
    for (y = 0; y < 32; y++)
    {
        do
        {
            gzgets(file, line, 80);
        } while ((line[0] != '0') && (line[0] != '1'));
        for (x = 0; x < 32; x++)
        {
            digit[x][y] = (line[x] == '0' ? 0 : 1);
        }
    }

    preprocess_char32x32(digit, 0, data->intensity);
    gzgets(file, line, 80);
    sscanf(line, " %c\n", &data->value);
}

int fscan_training_file(gzFile file, training_data_t *array)
{
    int i;

    i = 0;
    while (!gzeof(file))
    {
        fscan_training_digit(file, &array[i]);
        i++;
    }

    return i;
}

int read_training_file(const char *filename, training_data_t *array)
{
    int result;

    gzFile file;
    file = gzopen(filename, "r");
    result = fscan_training_file(file, array);
    gzclose(file);

    return result;
}
