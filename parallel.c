/* 
 *  Name: Coco Li
 *  UID: 905 917 242
 */

#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"



/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS]) {
    
    long count = num_rows*num_cols; //count is the pixels

    #pragma omp parallel for reduction(+:mean[:NUM_CHANNELS])
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
	    int comp = row*num_cols + col;
            for (int ch = 0; ch < NUM_CHANNELS; ch++){
                mean[ch] += img[comp][ch];
            }
        }
    }

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        mean[ch] /= count;
    }
}



/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count) {
    int tempMaxGray = 0;
    int tempMaxCount = 0;
   
    #pragma omp parallel for reduction (max:tempMaxGray)  
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++){
                int grayscaleVal = 0;
		for (int ch = 0; ch < NUM_CHANNELS; ch++){
		   grayscaleVal += img[row*num_cols + col][ch];
		}

		grayscaleVal /= NUM_CHANNELS;  //dividing the value by num_channels to get the grayscale val
                
		grayscale_img[row*num_cols + col][0] = grayscaleVal;
 		grayscale_img[row*num_cols + col][1] = grayscaleVal;
 		grayscale_img[row*num_cols + col][2] = grayscaleVal;
                
		if (grayscaleVal > tempMaxGray) {
                    tempMaxGray = grayscaleVal;
    		}
        }
    }

    /*put the tempMaxCount here because we can't have tempMaxCount as a reduction 
    * in the above loop bc maxcount depends on maxgray and we can't combine maxCount 
    * at the end when it relies on maxgray
    */  
    
     #pragma omp parallel for reduction (+:tempMaxCount)
     for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++){
            if (grayscale_img[row*num_cols + col][0] == tempMaxGray){
	       tempMaxCount+= 3;
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

    int kernel_norm;
    int conv_rows, conv_cols; 

    // compute kernel normalization factor
    kernel_norm = 0;
    for(int i = 0; i < kernel_size*kernel_size; i++) {
        kernel_norm += kernel[i];
    }

    // compute dimensions of convolved image
    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;

    // perform convolution

    #pragma omp parallel for
    for (int row = 0; row < conv_rows; row++) {  //each thread will be working on a row
       for (int col = 0; col < conv_cols; col++) {
          for (int ch = 0; ch < NUM_CHANNELS; ch++){
	     int tempAdd = 0;
             for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                   tempAdd += padded_img[(row+kernel_row)*num_cols + col+kernel_col][ch] * kernel[kernel_row*kernel_size + kernel_col];
                }
             }
             convolved_img[row*conv_cols + col][ch] = (tempAdd/kernel_norm);
          }
       }
   }



}

