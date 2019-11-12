//
// Created by Micheal Cowan on 9/7/19.
//

#include <DigitalSignalProcessing/Resample.h>
#include <common/BBP_Block.h>


// This function is a rationalle upsample
size_t upsample(RADIO_DATA_TYPE* in_block, size_t /*in_length*/, RADIO_DATA_TYPE* out_block, size_t out_length, RADIO_DATA_TYPE ratio)
{
    for(size_t out_index = 0, in_index = 0; out_index < out_length; in_index++)
    {
        out_block[out_index++] = in_block[in_index]; // Index J = 0;
        for(int j = 1; j < ceil(ratio) && out_index < out_length; j++)
        {
            out_block[out_index++] = in_block[in_index]+(ratio*j); // Index J > 0 until next major sample
        }
    }

    return out_length;
}

size_t upsample_fill_w_zeros(RADIO_DATA_TYPE* in_block, size_t in_length, RADIO_DATA_TYPE* out_block, size_t out_length, RADIO_DATA_TYPE ratio)
{
    for(size_t out_index = 0, in_index = 0; out_index < out_length && in_index < in_length; in_index++)
    {
        out_block[out_index++] = in_block[in_index]; // Index J > 0 until next major sample
        for(int j = 0; j < (ceil(ratio)-1) && out_index < out_length && in_index < in_length; j++)
        {
            out_block[out_index++] = 0;
        }
    }

    return out_length;
}

void decimate(BBP_Block* block, size_t decimate_by)
{
    size_t dst = 0;
    size_t src = 0;

    while(src+decimate_by < block->n_points)
    {
        block->points[dst++] = block->points[src+=decimate_by];
    }
    block->n_points=dst+1;
}

size_t decimateComplexArray(Complex_Array& block, size_t length, size_t decimate_by)
{
    size_t dst = 0;
    size_t src = 0;

    while(src < length)
    {
        block[dst].real(block[src].real());
        block[dst].imag(block[src].imag());
        src+=decimate_by;
        dst++;
    }

    for(size_t j = dst; j < length; j++)
    {
        block[dst].real(0);
        block[dst].imag(0);
    }

    return dst;
}



