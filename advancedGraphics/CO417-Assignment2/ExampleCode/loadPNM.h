#ifndef __LOADPNM_H__
#define __LOADPNM_H__

#include <iostream>

using namespace std;

unsigned char *
loadPNM( const char *filename,
	 unsigned int& width, unsigned int &height,
	 unsigned int &numComponents );

void
WritePNM( const char *filename,
	 unsigned int& width, unsigned int &height,
	 unsigned int &numComponents,
   unsigned char* imageData );
   

float *
loadPFM( const char *filename,
	 unsigned int& width, unsigned int &height,
	 unsigned int &numComponents );

void
WritePFM( const char *filename,
	 unsigned int& width, unsigned int &height,
	 unsigned int &numComponents,
   float* imageData );

   
#endif

