//
// Created by Micheal Cowan on 9/7/19.
//

#include "LowPassFilter.h"


LowPassFilter::LowPassFilter(RADIO_DATA_TYPE iCutOffFrequency, RADIO_DATA_TYPE iDeltaTime)
: output(0)
, ePow(1.0-exp(-iDeltaTime * iCutOffFrequency))
{
}

RADIO_DATA_TYPE LowPassFilter::update(RADIO_DATA_TYPE input)
{
    return output += (input - output) * ePow;
}

