//
// Created by Micheal Cowan on 9/7/19.
//

#include "FFT.h"



void fft(Complex_Array& block, size_t length)
{
    const size_t N = length;
    if (N <= 1) return;

    // divide
    Complex_Array even = block[std::slice(0, N/2, 2)];
    Complex_Array  odd = block[std::slice(1, N/2, 2)];

    // conquer
    fft(even, N/2);
    fft(odd, N/2);

    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar<RADIO_DATA_TYPE>(1.0, -2 * PI * k / N) * odd[k];
        block[k    ] = even[k] + t;
        block[k+N/2] = even[k] - t;
    }
}