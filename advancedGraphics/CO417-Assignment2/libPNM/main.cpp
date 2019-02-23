/*//////////////////////////////////////////////////////////////////////////
  Author: Abhijeet Ghosh
  Year: 2013
//////////////////////////////////////////////////////////////////////////*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "PNM.h"

//#define PI 3.14159265358979323
#define uint unsigned int

#include <iostream>

using namespace std;

unsigned int width;
unsigned int height;
unsigned int numComponents;

void CreateAndSavePFM(const char *image_out)
{
  width = 511; // set size of image to 511x511 pixels
  height = 511;
  numComponents = 3;
	
  float *img_out = new float [width*height*numComponents];

  for ( uint i = 0 ; i < height ; ++i ) // height
    {
      for ( uint j = 0 ; j < width ; ++j ) // width
	{
	  for ( uint k = 0 ; k < numComponents ; ++k ) // color channels - 3 for RGB images
	    {
	      uint index = i*width*numComponents + j*numComponents + k; //index within the image

	      //set image to white
	      img_out[index] = 1.0f; //RGB all set to white
								
	    }

	}
    }


  writePFM(image_out, width, height, numComponents, img_out);

}

void LoadAndSavePPM(const char *image_in, const char *image_out)
{
  unsigned char *img_in = loadPPM(image_in, &width, &height, &numComponents);
  unsigned char *img_out = new unsigned char [width*height*numComponents];

  for ( uint i = 0 ; i < height ; ++i ) // height
    {
      for ( uint j = 0 ; j < width ; ++j ) // width
	{
	  for ( uint k = 0 ; k < numComponents ; ++k ) // color channels - 3 for RGB images
	    {
	      uint index = i*width*numComponents + j*numComponents + k; //index within the image

	      img_out[index] = img_in[index]; //copy all color channels of each pixel
								
	    }

	}
    }


  writePPM(image_out, width, height, numComponents, img_out);

}

void LoadAndSavePFM(const char *image_in, const char *image_out)
{
  float *img_in = loadPFM(image_in, &width, &height, &numComponents);
  float *img_out = new float [width*height*numComponents];

  for ( uint i = 0 ; i < height ; ++i ) // height
    {
      for ( uint j = 0 ; j < width ; ++j ) // width
	{
	  for ( uint k = 0 ; k < numComponents ; ++k ) // color channels - 3 for RGB images
	    {
	      uint index = i*width*numComponents + j*numComponents + k; //index within the image

	      img_out[index] = img_in[index]; //copy all color channels of each pixel
							
	    }

	}
    }


  writePFM(image_out, width, height, numComponents, img_out);

}

void LoadPPMAndSavePFM(const char *image_in, const char *image_out)
{
  unsigned char *img_in = loadPPM(image_in, &width, &height, &numComponents);
  float *img_out = new float [width*height*numComponents];

  for ( uint i = 0 ; i < height ; ++i ) // height
    {
      for ( uint j = 0 ; j < width ; ++j ) // width
	{
	  for ( uint k = 0 ; k < numComponents ; ++k ) // color channels - 3 for RGB images
	    {
	      uint index = i*width*numComponents + j*numComponents + k; //index within the image

	      //typecast 0 - 255 values to the 0.0f -> 1.0f range 
	      img_out[index] = static_cast<float>(img_in[index])/255.0f; //typecast all color channels of each pixel
								
	    }

	}
    }


  writePFM(image_out, width, height, numComponents, img_out);

}

void LoadPFMAndSavePPM(const char *image_in, const char *image_out)
{
  float *img_in = loadPFM(image_in, &width, &height, &numComponents);
  unsigned char *img_out = new unsigned char [width*height*numComponents];

  for ( uint i = 0 ; i < height ; ++i ) // height
    {
      for ( uint j = 0 ; j < width ; ++j ) // width
	{
	  for ( uint k = 0 ; k < numComponents ; ++k ) // color channels - 3 for RGB images
	    {
	      uint index = i*width*numComponents + j*numComponents + k; //index within the image

	      //typecast 0.0f -> 1.0f values to the 0 - 255 range 
	      img_out[index] = static_cast<unsigned char>(img_in[index]*255.0f); //typecast all color channels of each pixel
								
	    }

	}
    }


  writePPM(image_out, width, height, numComponents, img_out);

}

int main(int argc, char** argv)
{
          
  cerr<<"main invoked: arguments - <image_out (.pfm)> "<<endl;
  cerr<<"main invoked: arguments - <image_in (.ppm)> <image_out (.ppm)> "<<endl;
  
  int count = argc;

  if(count < 2)
    {
      cout<<"Too few arguments! .... exiting."<<endl;
      return 0;
    }        

  if(count == 2)
    {  
      CreateAndSavePFM(argv[1]); //Creates and saves a PFM
    }
  else
    if(count == 3)
      {  
	//  LoadAndSavePPM(argv[1], argv[2]); //Loads and saves a PPM file
	//  LoadAndSavePFM(argv[1], argv[2]); //Loads and saves a PFM file
	//	LoadPPMAndSavePFM(argv[1], argv[2]); //Loads PPM and saves a PFM file
	LoadPFMAndSavePPM(argv[1], argv[2]); //Loads PFM and saves a PPM file
      }
 
  return 0;
}


