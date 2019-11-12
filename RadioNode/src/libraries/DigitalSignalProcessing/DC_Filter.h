//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_DC_FILTER_H
#define RADIONODE_DC_FILTER_H

#include <common/Common_Deffinitions.h>

// Original reference found here: https://www.dsprelated.com/freebooks/filters/DC_Blocker.html
class DC_Filter {
public:
    //constructors
    // R must be between 0.0 and 1.0
    // 0.995 for 44.1khz as a reference
    DC_Filter(RADIO_DATA_TYPE R = 0.995);

    //functions
    void update(RADIO_DATA_TYPE *input, RADIO_DATA_TYPE *output, size_t array_size);

private:
    RADIO_DATA_TYPE m_R;
};


#endif //RADIONODE_DC_FILTER_H
