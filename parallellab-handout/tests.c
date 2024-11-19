#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "sequential.h"
#include "parallel.h"
#include "utils.h"

#include "tests.h"



static inline void print_mean_pixel(const double arr[]) {
    int c;
    printf("[");
    for(c = 0; c < NUM_CHANNELS-1; c++) {
        printf("%f, ", arr[c]);
    }
    printf("%f]", arr[NUM_CHANNELS-1]);
}

static inline void print_image(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols) {
    int row, col, ch, i;
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++) {
            i = row*num_cols + col;
            printf("(");
            for (ch = 0; ch < NUM_CHANNELS-1; ch++) {
                printf("%d,", img[i][ch]);
            }
            printf("%d)\t", img[i][NUM_CHANNELS-1]);
        }
        printf("\n");
    }
}

static inline void print_output_image(const uint32_t img[][NUM_CHANNELS], int num_rows, int num_cols) {
    int row, col, ch, i;
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++) {
            i = row*num_cols + col;
            printf("(");
            for (ch = 0; ch < NUM_CHANNELS-1; ch++) {
                printf("%d,", img[i][ch]);
            }
            printf("%d)\t", img[i][NUM_CHANNELS-1]);
        }
        printf("\n");
    }
}

static inline int compare_images(const uint32_t sequential_img[][NUM_CHANNELS], const uint32_t parallel_img[][NUM_CHANNELS], int size) {
    int i, c;
    for (i = 0; i < size; i++) {
        for (c = 0; c < NUM_CHANNELS; c++) {
            if (sequential_img[i][c] != parallel_img[i][c]) {
                return 1;
            }
        }
    }
    return 0;
}



/* PHASE 1 TESTING */
double test_mean_pixel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols) {
    int ch;
    double seqMean[NUM_CHANNELS], parMean[NUM_CHANNELS];

    struct timespec start, finish;
    double seqDelay, parDelay;
    double speedup;

    printf("Running phase 1 on a %dx%d image...\n", num_rows, num_cols);

    clock_gettime(CLOCK_MONOTONIC, &start);
    mean_pixel_seq(img, num_rows, num_cols, seqMean);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Sequential version took %lf time units\n",seqDelay);

    clock_gettime(CLOCK_MONOTONIC, &start);
    mean_pixel_parallel(img, num_rows, num_cols, parMean);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Parallel version took %lf time units\n", parDelay);

    
    for (ch = 0; ch < NUM_CHANNELS; ch++) {
        if (fabs(seqMean[ch] - parMean[ch]) > EPS) {
            printf("Your phase 1 results are incorrect.\n");
            printf("The expected result was ");
            print_mean_pixel(seqMean);
            printf(".\nThe observed result was ");
            print_mean_pixel(parMean);
            printf(".\n\n");
            return -1;
        }
    }

    speedup = seqDelay / parDelay;
    printf("Your phase 1 results are correct for this run.\n");
    printf("Your phase 1 speedup was %fx.\n\n", speedup);
    return speedup;
}



/* PHASE 2 TESTING */
double test_greyscale(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols) {
    int error = 0;
    uint32_t (*sequential_grayscale_img)[NUM_CHANNELS], (*parallel_grayscale_img)[NUM_CHANNELS];
    uint8_t sequential_max_gray, parallel_max_gray;
    uint32_t sequential_max_count, parallel_max_count;

    double speedup;
    struct timespec start, finish;
    double seqDelay, parDelay;

    // dynamically allocate grayscale images
    sequential_grayscale_img = malloc(sizeof(uint32_t) * num_rows*num_cols*NUM_CHANNELS);
    parallel_grayscale_img = malloc(sizeof(uint32_t) * num_rows*num_cols*NUM_CHANNELS);

    printf("Running phase 2 on a %dx%d image...\n", num_rows, num_cols);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_seq(img, num_rows, num_cols, sequential_grayscale_img, &sequential_max_gray, &sequential_max_count);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Sequential version took %lf time units\n",seqDelay);

    clock_gettime(CLOCK_MONOTONIC, &start);
    grayscale_parallel(img, num_rows, num_cols, parallel_grayscale_img, &parallel_max_gray, &parallel_max_count);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Parallel version took %lf time units\n", parDelay);

    error = compare_images(sequential_grayscale_img, parallel_grayscale_img, num_rows*num_cols);
    if (sequential_max_count != parallel_max_count || sequential_max_gray != parallel_max_gray) {
        error += 2;
    }

    if (error%2 == 1) {
        printf("Your phase 2 image results are incorrect.\n\n");
    }

    if (error >= 2) {
        printf("Your phase 2 max gray results are incorrect.\n");
        printf("The expected max gray value was %d, appearing %d times.\n", sequential_max_gray, sequential_max_count);
        printf("The observed max gray value was %d, appearing %d times.\n\n", parallel_max_gray, parallel_max_count);
    }

    free(sequential_grayscale_img);
    free(parallel_grayscale_img);

    if (error != 0) {
        return -1;
    }

    speedup = seqDelay / parDelay;
    printf("Your phase 2 results are correct for this run.\n");
    printf("Your phase 2 speedup was %fx.\n\n", speedup);
    return speedup;
}



/* PHASE 3 TESTING */
double test_convolution(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size) {
    int error = 0;
    int padding, pad_rows, pad_cols;
    int row, col, ch;

    uint8_t (*padded_img)[NUM_CHANNELS];
    uint32_t (*sequential_conv_img)[NUM_CHANNELS], (*parallel_conv_img)[NUM_CHANNELS];

    double speedup;
    struct timespec start, finish;
    double seqDelay, parDelay;

    assert(kernel_size % 2 == 1);
    padding = (kernel_size-1) / 2;
    pad_rows = num_rows + 2*padding;
    pad_cols = num_cols + 2*padding;

    // dynamically allocate convolved and padded images
    sequential_conv_img = malloc(sizeof(uint32_t) * num_rows*num_cols*NUM_CHANNELS);
    parallel_conv_img = malloc(sizeof(uint32_t) * num_rows*num_cols*NUM_CHANNELS);

    padded_img = malloc(sizeof(uint8_t) * pad_rows*pad_cols*NUM_CHANNELS);
    for (row=0; row<num_rows; row++) {
        for (col=0; col<num_cols; col++) {
            for (ch=0; ch<NUM_CHANNELS; ch++) {
                padded_img[(row+padding)*pad_cols + col+padding][ch] = img[row*num_cols + col][ch];
            }
        }
    }

    printf("Running phase 3 on a %dx%d image with a %dx%d kernel...\n", num_rows,num_cols,kernel_size,kernel_size);

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution_seq(padded_img, pad_rows, pad_cols, kernel, kernel_size, sequential_conv_img);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    seqDelay = (finish.tv_sec - start.tv_sec);
    seqDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Sequential version took %lf time units\n",seqDelay);

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution_parallel(padded_img, pad_rows, pad_cols, kernel, kernel_size, parallel_conv_img);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    parDelay = (finish.tv_sec - start.tv_sec);
    parDelay += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Parallel version took %lf time units\n", parDelay);

    error = compare_images(sequential_conv_img, parallel_conv_img, num_rows*num_cols);

    free(sequential_conv_img);
    free(parallel_conv_img);
    free(padded_img);

    if (error != 0) {
        printf("Your phase 3 image results are incorrect.\n\n");
        return -1;
    }

    speedup = seqDelay / parDelay;
    printf("Your phase 3 results are correct for this run.\n");
    printf("Your phase 3 speedup was %fx.\n\n", speedup);
    return speedup;
}