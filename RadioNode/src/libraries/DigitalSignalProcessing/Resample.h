//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_RESAMPLE_H
#define RADIONODE_RESAMPLE_H

#include <common/Common_Deffinitions.h>
#include <common/BBP_Block.h>


size_t upsample(RADIO_DATA_TYPE* in_block, size_t in_length, RADIO_DATA_TYPE* out_block, size_t out_length, RADIO_DATA_TYPE ratio);

// This function is a raw upsample
template<class data_type>
size_t raw_upsample(data_type* in_block, size_t in_length, data_type* out_block, size_t out_length, data_type ratio)
{
    for(size_t out_index = 0, in_index = 0; out_index < out_length && in_index < in_length; in_index++)
    {
        for(int j = 0; (j < floor(ratio)) && (out_index < out_length); j++)
        {
            out_block[out_index++] = in_block[in_index]; // Index J > 0 until next major sample
        }
    }

    return out_length;
}
size_t upsample_fill_w_zeros(RADIO_DATA_TYPE* in_block, size_t in_length, RADIO_DATA_TYPE* out_block, size_t out_length, RADIO_DATA_TYPE ratio);


void decimate(BBP_Block* block, size_t decimate_by = 2);

template<class data_type>
size_t decimate(data_type * block, size_t length, size_t decimate_by = 2)
{
    size_t dst = 0;
    size_t src = 0;

    while(src < length)
    {
        block[dst] = block[src];
        src+=decimate_by;
        dst++;
    }

    for(size_t j = dst; j < length; j++)
    {
        block[dst] = 0;
    }

    return dst;
}

size_t decimateComplexArray(Complex_Array& block, size_t length, size_t decimate_by = 2);

#endif //RADIONODE_RESAMPLE_H
