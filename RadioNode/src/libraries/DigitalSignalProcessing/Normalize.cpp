//
// Created by Micheal Cowan on 9/7/19.
//

#include "Normalize.h"



RADIO_DATA_TYPE normalize(RADIO_DATA_TYPE* block, size_t len, RADIO_DATA_TYPE prev_max)
{
    size_t max = prev_max;
    for(size_t i = 0; i < len; i++)
    {
        if(max < abs(block[i]))
        {
            max = abs(block[i]);
        }
    }

    for(size_t i = 0; i < len; i++)
    {
        block[i] = (block[i] / max)*2.0 - 1.0;
    }

    return max;
}


RADIO_DATA_TYPE normalize(BBP_Block* block, RADIO_DATA_TYPE prev_max)
{
    size_t max = prev_max;
    for(size_t i = 0; i < block->number_of_points(); i++)
    {
        if(max < abs(block->points[i].real()))
        {
            max = abs(block->points[i].real());
        }

        if(max < abs(block->points[i].imag()))
        {
            max = abs(block->points[i].imag());
        }
    }

    for(size_t i = 0; i < block->number_of_points(); i++)
    {
        block->points[i].real((block->points[i].real() / max)*2.0 - 1.0);
        block->points[i].imag((block->points[i].imag() / max)*2.0 - 1.0);
    }

    return max;
}

