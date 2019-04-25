#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include <arm_neon.h>

using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  //double color;

  uint16_t scalar1 = (uint16_t) (.144*256);				//make scalar 1 into integer
  uint16_t scalar2 = (uint16_t) (.587*256);				//make scalar 2 into integer
  uint16_t scalar3 = (uint16_t) (.299*256);				//make scalar 3 into integer

  // Convert to grayscale
  for (int i=0; i<img.rows-1; i++) {
    for (int j=0; j<img.cols-1; j += 8) {
     /* color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
     */
    	uint16x8_t scalar1_vector = vmovq_n_u16(scalar1);	 	//put scalar 1 into a vector
	uint16x8_t scalar2_vector = vmovq_n_u16(scalar2); 		//put scalar 2 into a vector
	uint16x8_t scalar3_vector = vmovq_n_u16(scalar3); 		//put scalar 3 into a vector
	uint8x8x3_t vector = vld3_u8(&img.data[STEP0*i + STEP1*j]); 	//first 8 bits of RGB value; 3 different vectors
	
	uint16x8_t red = vmovl_u8(vector.val[0]); 			//first 8 values of red
	uint16x8_t blue = vmovl_u8(vector.val[1]); 			//first 8 values of blue
	uint16x8_t green = vmovl_u8(vector.val[2]); 			//first 8 values of green
	red = vmulq_u16(red, scalar1_vector); 				//multiply red values by scalar 1
        uint8x8_t red_shifted = vqshrn_n_u16(red, 8);			//shift to undo 255 
	blue = vmulq_u16(blue, scalar2_vector); 			//multiply blue values by scalar 2
	uint8x8_t blue_shifted = vqshrn_n_u16(blue, 8);			//shift to undo 255
        green = vmulq_u16(green, scalar3_vector); 			//multiply red values by scalar 3
	uint8x8_t green_shifted = vqshrn_n_u16(green, 8);		//shift to undo 255
        uint8x8_t total = vadd_u8(red_shifted, blue_shifted);		//add colors 
        total = vadd_u8(total, green_shifted);				//add colors
	vst1_u8(&img_gray_out.data[IMG_WIDTH*i + j], total); 		//load into data out
    }
  }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  Mat img_outx = img_gray.clone();
  Mat img_outy = img_gray.clone();

  // Apply Sobel filter to black & white image
  //unsigned short sobel;
  
  uint16_t scalar1 = 2;
  uint16_t comparison1 = 255;	
  //load scalars
  uint16x8_t scalar_vector = vld1q_dup_u16(&scalar1);
  //load comparison
  uint16x8_t comparison_vector = vld1q_dup_u16(&comparison1);


  // Calculate the x convolution
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=1; j<img_gray.cols-1; j+=8) {
     /* sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobel = (sobel > 255) ? 255 : sobel;
      img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
  */
	//load elements into vectors
        uint8x8_t vec1 = vld1_u8(&img_gray.data[IMG_WIDTH*(i-1) + (j-1)]);
	uint8x8_t vec2 = vld1_u8(&img_gray.data[IMG_WIDTH*(i+1) + (j-1)]);
	uint8x8_t vec3 = vld1_u8(&img_gray.data[IMG_WIDTH*(i-1) + (j)]);
	uint8x8_t vec4 = vld1_u8(&img_gray.data[IMG_WIDTH*(i+1) + (j)]);
	uint8x8_t vec5 = vld1_u8(&img_gray.data[IMG_WIDTH*(i-1) + (j+1)]);
	uint8x8_t vec6 = vld1_u8(&img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);
	
	//make 16 bit to ensure no overflow
	uint16x8_t vec3_16bit = vmovl_u8(vec3);
	uint16x8_t vec4_16bit = vmovl_u8(vec4);

	//do multiplication
	vec3_16bit = vmulq_u16(vec3_16bit, scalar_vector);
	vec4_16bit = vmulq_u16(vec4_16bit, scalar_vector);
	
	//arithmetic
	uint16x8_t additions = vaddw_u8(vec3_16bit, vec1);
	additions = vaddw_u8(additions, vec5);
	uint16x8_t subtractions = vaddw_u8(vec4_16bit, vec2);
	subtractions = vaddw_u8(subtractions, vec6);

	//absolute value
	uint16x8_t total = vabdq_u16(additions, subtractions);

	//comparison total with max
	uint16x8_t comparison = vcgtq_u16(total, comparison_vector);
	total = vbslq_u16(comparison, comparison_vector, total); 
	uint8x8_t total_8bit = vqmovn_u16(total);
	//load into data out
	vst1_u8(&img_outx.data[IMG_WIDTH*i + j], total_8bit);
	
	}
  }

  // Calc the y convolution
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=1; j<img_gray.cols-1; j+=8) {
     /*sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

     sobel = (sobel > 255) ? 255 : sobel;

     img_outy.data[IMG_WIDTH*(i) + j] = sobel; */
	
	//load elements into vectors
	uint8x8_t vec1 = vld1_u8(&img_gray.data[IMG_WIDTH*(i-1) + (j-1)]);
	uint8x8_t vec2 = vld1_u8(&img_gray.data[IMG_WIDTH*(i-1) + (j+1)]);
	uint8x8_t vec3 = vld1_u8(&img_gray.data[IMG_WIDTH*(i) + (j-1)]);
	uint8x8_t vec4 = vld1_u8(&img_gray.data[IMG_WIDTH*(i) + (j+1)]);
	uint8x8_t vec5 = vld1_u8(&img_gray.data[IMG_WIDTH*(i+1) + (j-1)]);
	uint8x8_t vec6 = vld1_u8(&img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

		
	//make 16 bit to ensure no overflow
	uint16x8_t vec3_16bit = vmovl_u8(vec3);
	uint16x8_t vec4_16bit = vmovl_u8(vec4);

	//do multiplication
	vec3_16bit = vmulq_u16(vec3_16bit, scalar_vector);
	vec4_16bit = vmulq_u16(vec4_16bit, scalar_vector);
	
	//arithmetic
	uint16x8_t additions = vaddw_u8(vec3_16bit, vec1);
	additions = vaddw_u8(additions, vec5);
	uint16x8_t subtractions = vaddw_u8(vec4_16bit, vec2);
	subtractions = vaddw_u8(subtractions, vec6);

	//absolute value
	uint16x8_t total = vabdq_u16(additions, subtractions);

	//comparison total with max
	uint16x8_t comparison = vcgtq_u16(total, comparison_vector);
	total = vbslq_u16(comparison, comparison_vector, total); 
	uint8x8_t total_8bit = vqmovn_u16(total);

	//load into data out
	vst1_u8(&img_outy.data[IMG_WIDTH*i + j], total_8bit);
	

    }
  }

  // Combine the two convolutions into the output image
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=1; j<img_gray.cols-1; j+=8) {
      /*sobel = img_outx.data[IMG_WIDTH*(i) + j] +
      img_outy.data[IMG_WIDTH*(i) + j];
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel; */

	//load array into vectors
	uint8x8_t vec1 = vld1_u8(&img_outx.data[IMG_WIDTH*(i) + (j)]);
	uint8x8_t vec2 = vld1_u8(&img_outy.data[IMG_WIDTH*(i) + (j)]);
	uint16x8_t total = vaddl_u8(vec1, vec2);
	//comparison total with max
	uint16x8_t comparison = vcgtq_u16(total, comparison_vector);
	total = vbslq_u16(comparison, comparison_vector, total);
	//load into data out
	uint8x8_t total_8bit = vqmovn_u16(total);
	vst1_u8(&img_sobel_out.data[IMG_WIDTH*i + j], total_8bit);
    }
  }
}
