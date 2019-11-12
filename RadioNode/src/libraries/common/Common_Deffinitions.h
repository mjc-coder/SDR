//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_COMMON_DEFFINITIONS_H
#define RADIONODE_COMMON_DEFFINITIONS_H

#include <stdlib.h>
#include <cmath>
#include <complex>
#include <valarray>

typedef double RADIO_DATA_TYPE;

typedef std::complex<RADIO_DATA_TYPE> Complex;
typedef std::valarray<Complex> Complex_Array;

const RADIO_DATA_TYPE PI = 3.141592653589793238460;

#define MHZ_TO_HZ(freq) (freq * 1000000)


// Block Read Size -- 65536 Samples / 32768 Points / 131072 bytes
#define BLOCK_READ_SIZE 240000/2

#endif //RADIONODE_COMMON_DEFFINITIONS_H
