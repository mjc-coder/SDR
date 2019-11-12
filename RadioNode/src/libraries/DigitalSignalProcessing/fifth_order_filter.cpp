//
// Created by Micheal Cowan on 9/7/19.
//

#include "fifth_order_filter.h"


fifth_order_filter::fifth_order_filter()
: r_a(0)
, r_b(0)
, r_c(0)
, r_d(0)
, r_e(0)
, r_f(0)
, i_a(0)
, i_b(0)
, i_c(0)
, i_d(0)
, i_e(0)
, i_f(0)
, a(1)
, b(5)
, c(10)
, d(11)
{
}

void fifth_order_filter::decimate(BBP_Block* block, size_t decimate_by)
{
    /* a downsample should improve resolution, so don't fully shift */
    for(size_t dec = 2; dec <= decimate_by; dec*=2)
    {
        block->points[0].real(((a*(r_a+r_f)) + ((r_b + r_e)*b) + ((r_c+r_d)*c)) / d); // divide by 16
        block->points[0].imag(((a*(i_a+i_f)) + ((i_b + i_e)*b) + ((i_c+i_d)*c)) / d); // divide by 16
        size_t dst = 2;
        for (size_t i = 2; i < block->number_of_points(); i += 2)
        {
            dst = i/2;
            r_a = r_c;
            r_b = r_d;
            r_c = r_e;
            r_d = r_f;
            r_e = block->points[i - 2].real();
            r_f = block->points[i].real();
            block->points[dst].real( ((a*(r_a+r_f)) + ((r_b + r_e)*b) + ((r_c+r_d)*c)) / d );

            i_a = i_c;
            i_b = i_d;
            i_c = i_e;
            i_d = i_f;
            i_e = block->points[i - 2].imag();
            i_f = block->points[i].imag();
            block->points[dst].imag( ((a*(i_a+i_f)) + ((i_b + i_e)*b) + ((i_c+i_d)*c)) / d ); // divide by 16
        }
        block->n_points = dst+1;
    }
}


size_t fifth_order_filter::decimate(RADIO_DATA_TYPE* data, size_t length, size_t decimate_by)
{
    size_t dst = 0;
    /* a downsample should improve resolution, so don't fully shift */
    for(size_t dec = 2; dec <= decimate_by; dec*=2)
    {
        dst = 2;
        data[0] = ((a*(r_a + r_f)) + ((r_b + r_e) * b) + ((r_c + r_d) * c)) / d; // divide by 16
        for (size_t i = 2; i < length; i += 2)
        {
            r_a = r_c;
            r_b = r_d;
            r_c = r_e;
            r_d = r_f;
            r_e = data[i - 2];
            r_f = data[i];
            dst = i/2;
            data[dst] = ((a*(r_a + r_f)) + ((r_b + r_e) * b) + ((r_c + r_d) * c)) / d;
        }
    }

    return length/decimate_by;
}
