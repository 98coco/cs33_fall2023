#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "tests.h"
#include "utils.h"



static const uint32_t gblur_3x3_kernel[] = {
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};

static const uint32_t gblur_5x5_kernel[] = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
};

// Display usage info
static void usage(char *cmd) {
    printf("Usage: %s [-h] [-p <n>] [-r <n>] [-c <n>]\n", cmd);
    printf("  -h        Print this message.\n");
    printf("  -p <n>    Test only phase n.\n");
    printf("  -r <n>    Generate image with n rows. Use random number of rows if n is 0.\n");
    printf("  -c <n>    Generate image with n columns. Use random number of columns if n is 0.\n");
    exit(1);
}

int main(int argc, char* argv[])
{
    char c;
    int phase;

    int num_rows, num_cols, log_dim_range;
    int rgb_val1, rgb_val2, rgb_min[NUM_CHANNELS], rgb_range[NUM_CHANNELS];

    uint8_t (*test_img)[NUM_CHANNELS];
    int row, col, ch, rgb_val;


    /* parse command line args */
    phase = 0;
    num_rows = 0, num_cols = 0;
    while ((c = getopt(argc, argv, "hp:r:c:")) != -1) {
        switch (c) {
            case 'h': /* help */
                usage(argv[0]);
                break;
            case 'p': /* select phase to test */
                phase = atoi(optarg);
                if (phase < 1 || phase > 3) usage(argv[0]);
                break;
            case 'r': /* select image rows */
                num_rows = atoi(optarg);
                if (num_rows < 0 || num_rows > (1<<MAX_LOG_DIM)) usage(argv[0]);
                break;
            case 'c': /* select image cols */
                num_cols = atoi(optarg);
                if (num_cols < 0 || num_cols > (1<<MAX_LOG_DIM)) usage(argv[0]);
                break;
            default:
                usage(argv[0]);
        }
    }
    

    /* randomly generate an image to test */
    srand(time(NULL));
    
    log_dim_range = MAX_LOG_DIM - MIN_LOG_DIM + 1;
    if (num_rows == 0) {
        num_rows = 1 << (MIN_LOG_DIM + rand() % log_dim_range);
    }
    if (num_cols == 0) {
        num_cols = 1 << (MIN_LOG_DIM + rand() % log_dim_range);
    }
    
    for (ch = 0; ch < NUM_CHANNELS; ch++) {
        rgb_range[ch] = -1;
        while(rgb_range[ch] < MIN_VALUE_RANGE) {
            rgb_val1 = rand()%VALUE_RANGE;
            rgb_val2 = rand()%VALUE_RANGE;
            rgb_range[ch] = 1 + abs(rgb_val2 - rgb_val1);
        }
        rgb_min[ch] = MIN(rgb_val1, rgb_val2);
    }

    printf("Use CTRL-C to exit if necessary.\n");
    printf("Make sure to take care of possible data races!\n\n");
    printf("Generating a %dx%d image...\n\n", num_rows, num_cols);

    test_img = malloc(sizeof(uint8_t) * num_rows*num_cols*NUM_CHANNELS);
    for (row=0; row<num_rows; row++) {
        for (col=0; col<num_cols; col++) {
            for (ch=0; ch<NUM_CHANNELS; ch++) {
                rgb_val = rgb_min[ch] + (rand()%rgb_range[ch]);
                test_img[row*num_cols + col][ch] = rgb_val;
            }
        }
    }


    /* testing */
    switch(phase) {
        case 1:
            test_mean_pixel(test_img, num_rows, num_cols);
            break;
        case 2:
            test_greyscale(test_img, num_rows, num_cols);
            break;
        case 3:
            test_convolution(test_img, num_rows, num_cols, gblur_3x3_kernel, 3);
            test_convolution(test_img, num_rows, num_cols, gblur_5x5_kernel, 5);
            break;
        default:
            test_mean_pixel(test_img, num_rows, num_cols);
            test_greyscale(test_img, num_rows, num_cols);
            test_convolution(test_img, num_rows, num_cols, gblur_3x3_kernel, 3);
            test_convolution(test_img, num_rows, num_cols, gblur_5x5_kernel, 5);
    }


    /* cleanup */
    free(test_img);

    return 0;
}
