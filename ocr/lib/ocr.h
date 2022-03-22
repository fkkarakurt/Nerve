#ifndef OCR_H
#define OCR_H

#include <zlib.h>
#include "nerveNet.h"

typedef struct
{
    float intensity[8][8];
    char value;
} training_data_t;

void rescale_intensity8x8(float intensity[8][8]);
void preprocess_char32x32(char c[32][32], int, float intensity[8][8]);
void preprocess_char128x128(char c[128][128], int, float intensity[8][8]);

void fscan_training_digit(gzFile, training_data_t *);
int fscan_training_file(gzFile, training_data_t *);
int read_training_file(const char *, training_data_t *);

float get_average_error(network_t *, training_data_t *, int);
void train_all(network_t *, training_data_t *, int);
void train_batch(network_t *, training_data_t *, int);
void train_randomly(network_t *, training_data_t *, int, int);
void minimize_with_train_all(network_t *, training_data_t *, int,
                             training_data_t *, int, int, float, int, int);
void minimize_with_train_batch(network_t *, training_data_t *, int,
                               training_data_t *, int, int, float, int, int);
void minimize_with_train_randomly(network_t *, training_data_t *, int,
                                  training_data_t *, int, int, float, int,
                                  int);

#endif
