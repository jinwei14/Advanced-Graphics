#ifndef __LOADPNM_H__
#define __LOADPNM_H__

#include <stdio.h>
#include <stdlib.h>

unsigned char *
loadPPM( const char *filename,
	 unsigned int *width, unsigned int *height,
	 unsigned int *numComponents );

void
writePPM( const char *filename,
          unsigned int width, unsigned int height,
          unsigned int numComponents,
          unsigned char* imageData );
   

float *
loadPFM( const char *filename,
	 unsigned int *width, unsigned int *height,
	 unsigned int *numComponents );

void
writePFM( const char *filename,
          unsigned int width, unsigned int height,
          unsigned int numComponents,
          float* imageData );

   
#endif

