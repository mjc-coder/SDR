//
// Created by Micheal Cowan on 9/7/19.
//

#include "FirstOrderDigitalIntegration.h"


FirstOrderDigitalIntegration::FirstOrderDigitalIntegration(RADIO_DATA_TYPE sampleRate)
        : tStep(1.0f/sampleRate)
{

}

void FirstOrderDigitalIntegration::integrate(RADIO_DATA_TYPE* array, size_t length)
{
    for(size_t i = 1; i < length; i++)
    {
        array[i] = array[i - 1] + tStep * array[i];
    }
}

