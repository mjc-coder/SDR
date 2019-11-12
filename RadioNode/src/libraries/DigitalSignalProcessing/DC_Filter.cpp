//
// Created by Micheal Cowan on 9/7/19.
//

#include "DC_Filter.h"


DC_Filter::DC_Filter(RADIO_DATA_TYPE R)
        : m_R(R) {
}


void DC_Filter::update(RADIO_DATA_TYPE *input, RADIO_DATA_TYPE *output, size_t array_size) {
    output[0] = input[0];

    // y(n) = x(n) - x(n-1) + R * y(n-1)
    for (size_t i = 1; i < array_size; i++) {
        output[i] = input[i] - input[i - 1] + m_R * output[i - 1];
    }
}