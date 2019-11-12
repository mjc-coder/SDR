//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_LOWPASSFILTER_H
#define RADIONODE_LOWPASSFILTER_H

#include <common/Common_Deffinitions.h>

// Original reference found here: https://github.com/overlord1123/LowPassFilter/blob/master/
class LowPassFilter
{
public:
    //constructors
    LowPassFilter(RADIO_DATA_TYPE iCutOffFrequency, RADIO_DATA_TYPE iDeltaTime);
    //functions
    RADIO_DATA_TYPE update(RADIO_DATA_TYPE input);

private:
    RADIO_DATA_TYPE output;
    RADIO_DATA_TYPE ePow;
};


#endif //RADIONODE_LOWPASSFILTER_H
