#ifndef TESTS_H
#define TESTS_H

#include <stdint.h>
#include "utils.h"

double test_mean_pixel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols);
double test_greyscale(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols);
double test_convolution(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size);

#endif