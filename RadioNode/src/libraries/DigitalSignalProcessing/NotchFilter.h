//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_NOTCHFILTER_H
#define RADIONODE_NOTCHFILTER_H

#include <stdint.h>
#include <math.h>
#include <common/Common_Deffinitions.h>

class NotchFilter
{
public:
    NotchFilter(RADIO_DATA_TYPE notch_bw, RADIO_DATA_TYPE center_freq);

    void filter(RADIO_DATA_TYPE* block_in, size_t length);

private:
    RADIO_DATA_TYPE R;
    RADIO_DATA_TYPE K;
    RADIO_DATA_TYPE a0;
    RADIO_DATA_TYPE a1;
    RADIO_DATA_TYPE a2;
    RADIO_DATA_TYPE b1;
    RADIO_DATA_TYPE b2;
};


#endif //RADIONODE_NOTCHFILTER_H
