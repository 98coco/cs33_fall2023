/* 
 *  Name: [YOUR NAME HERE]
 *  UID: [YOUR UID HERE]
 */

#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"



/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
//void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS]) {
//    int row, col, ch;
//    long count = num_cols*num_rows;
//
//#pragma omp parallel for private(row, col, ch) reduction(+:mean[:NUM_CHANNELS])
//    for (row = 0; row < num_rows; row++) {
//        for (col = 0; col < num_cols; col++) {
//            int comp = row*num_cols + col;
//            for (ch = 0; ch < NUM_CHANNELS; ch++){
//                mean[ch] += img[comp][ch];
//            }
//        }
//    }
//
//    for (ch = 0; ch < NUM_CHANNELS; ch++) {
//        mean[ch] /= count;
//    }
//}

void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS]) {
    int row, col;
    long count = num_rows * num_cols;
    double mean1 = 0;
    double mean2 = 0;
    double mean3 = 0;
    
#pragma omp parallel for private(row, col) reduction(+:mean1,mean2,mean3) 
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++) {
            mean1 += img[row*num_cols + col][0];
            mean2 += img[row*num_cols + col][1];
            mean3 += img[row*num_cols + col][2];
        }
    }

    mean[0] = mean1/count;
    mean[1] = mean2/count;
    mean[2] = mean3/count;
}
    


/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count) {
    int row, col, ch;
    int tempMaxGray = 0, tempMaxCount = 0;

    #pragma omp parallel for private(col,ch) reduction(max:tempMaxGray)
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++){
                int grayscaleVal = 0; //reintialize everytime we are in a new element
                for (ch = 0; ch < NUM_CHANNELS; ch++) {
                    grayscaleVal += img[row*num_cols + col][ch]; //adding all the rgb in one pixel to compute the grayscaleVal
                }
                grayscaleVal = grayscaleVal/NUM_CHANNELS;
                grayscale_img[row*num_cols + col][0] = grayscaleVal;
                grayscale_img[row*num_cols + col][1] = grayscaleVal;
                grayscale_img[row*num_cols + col][2] = grayscaleVal;
            if (grayscaleVal > tempMaxGray){
                tempMaxGray = grayscaleVal;
            }
        }
    }
    
    //have seperate loops for maxCount and MaxGray so we can have threads working in parallel to find the max gray value and how find how man times it
    //is found in each elem
    
   #pragma omp parallel for private (col) reduction(max:tempMaxCount)
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++){
            if (grayscale_img[row*num_cols + col][0] == tempMaxGray){
                tempMaxCount+=3;
            }
        }
    }
    
    *max_gray = tempMaxGray;
    *max_count = tempMaxCount;
}



/*
 *  PHASE 3: perform convolution on image
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[][NUM_CHANNELS]) {
    int row, col, ch, kernel_row, kernel_col;
    int kernel_norm, i;
    int conv_rows, conv_cols;

    // compute kernel normalization factor

    kernel_norm = 0;
    for(i = 0; i < kernel_size*kernel_size; i++) {
        kernel_norm += kernel[i];
    }
    // compute dimensions of convolved image
    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;
    // perform convolution
    //no need to privatize convolved_img because each thread is working on a differnt row respectively
    #pragma omp parallel for private (row, col, ch, kernel_col,kernel_row)
    for (row = 0; row < conv_rows; row++) {
        for (col = 0; col < conv_cols; col++) {
            for (ch = 0; ch < NUM_CHANNELS; ch++) {
                int temp = 0;
                
                for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        temp += padded_img[(row+kernel_row)*num_cols + col+kernel_col][ch] * kernel[kernel_row*kernel_size + kernel_col];
                    }
                }
                convolved_img[row*conv_cols + col][ch] = (temp/kernel_norm);
            }
        }
    }
    
}

