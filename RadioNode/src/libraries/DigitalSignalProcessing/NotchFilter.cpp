//
// Created by Micheal Cowan on 9/7/19.
//

#include "NotchFilter.h"



NotchFilter::NotchFilter(RADIO_DATA_TYPE notch_bw, RADIO_DATA_TYPE center_freq)
        : R( 1.0 * notch_bw)
        , K( (1-2*R*cos(2.0*3.14159*center_freq) + R*R) / (2-2*cos(2*3.14159*center_freq)))
        , a0(K)
        , a1(-2.0*K*cos(2.0*3.14159*center_freq))
        , a2(K)
        , b1(2.0*R*cos(2.0*3.14159*center_freq))
        , b2(-(R*R))
{
}


void NotchFilter::filter(RADIO_DATA_TYPE* block_in, size_t length)
{
    static RADIO_DATA_TYPE x_2 = 0.0f;                    // delayed x, y samples
    static RADIO_DATA_TYPE x_1 = 0.0f;
    static RADIO_DATA_TYPE y_2 = 0.0f;
    static RADIO_DATA_TYPE y_1 = 0.0f;
    RADIO_DATA_TYPE delay = 0;
    for (size_t i = 2; i < length; i++)
    {
        delay = block_in[i];
        block_in[i] = a0*block_in[i] + a1*x_1 + a2*x_2  // IIR difference equation
                      + b1*y_1 + b2*y_2;
        x_2 = x_1;                              // shift delayed x, y samples
        x_1 = delay;
        y_2 = y_1;
        y_1 = block_in[i];
    }
}

