//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_FIRSTORDERDIGITALINTEGRATION_H
#define RADIONODE_FIRSTORDERDIGITALINTEGRATION_H

#include <common/Common_Deffinitions.h>

class FirstOrderDigitalIntegration {
public:
    FirstOrderDigitalIntegration(RADIO_DATA_TYPE sampleRate);

    void integrate(RADIO_DATA_TYPE *array, size_t length);

private:
    RADIO_DATA_TYPE tStep;
};


#endif //RADIONODE_FIRSTORDERDIGITALINTEGRATION_H
