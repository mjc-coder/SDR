/// @file SDR/RadioNode/src/libraries/DigitalSignalProcessing/LowPassFilter.h


#ifndef RADIONODE_LOWPASSFILTER_H
#define RADIONODE_LOWPASSFILTER_H

#include <common/Common_Deffinitions.h>

/// \brief A simple Low Pass Filter IIR
/// \details Original reference found here: https://github.com/overlord1123/LowPassFilter/blob/master/
class LowPassFilter
{
public:
    /// \brief Constructor
    /// \param iCutOffFrequency Frequency in Hertz
    /// \param iDeltaTime Time between samples
    LowPassFilter(RADIO_DATA_TYPE iCutOffFrequency, RADIO_DATA_TYPE iDeltaTime);
    //functions

    /// \brief Takes in a sample, and calculates the filtered response
    /// \param input Input data point
    /// \return Output data point
    RADIO_DATA_TYPE update(RADIO_DATA_TYPE input);

private:
    RADIO_DATA_TYPE output; ///< Last output value
    RADIO_DATA_TYPE ePow;   ///< Calculated Exponential value.
};


#endif //RADIONODE_LOWPASSFILTER_H
