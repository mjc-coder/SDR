//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_FIFTH_ORDER_FILTER_H
#define RADIONODE_FIFTH_ORDER_FILTER_H

#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <stdint.h>

class fifth_order_filter {
public:
    fifth_order_filter();

    ~fifth_order_filter() {};

    void decimate(BBP_Block* block, size_t decimate_by = 2);

    size_t decimate(RADIO_DATA_TYPE* data, size_t length, size_t decimate_by = 2);

private:
    RADIO_DATA_TYPE r_a;
    RADIO_DATA_TYPE r_b;
    RADIO_DATA_TYPE r_c;
    RADIO_DATA_TYPE r_d;
    RADIO_DATA_TYPE r_e;
    RADIO_DATA_TYPE r_f;
    RADIO_DATA_TYPE i_a;
    RADIO_DATA_TYPE i_b;
    RADIO_DATA_TYPE i_c;
    RADIO_DATA_TYPE i_d;
    RADIO_DATA_TYPE i_e;
    RADIO_DATA_TYPE i_f;

    RADIO_DATA_TYPE a, b, c, d;
};


#endif //RADIONODE_FIFTH_ORDER_FILTER_H
