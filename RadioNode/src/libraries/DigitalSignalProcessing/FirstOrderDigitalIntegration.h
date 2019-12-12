/// @file SDR/RadioNode/src/libraries/DigitalSignalProcessing/FirstOrderDigitalIntegration.h


#ifndef RADIONODE_FIRSTORDERDIGITALINTEGRATION_H
#define RADIONODE_FIRSTORDERDIGITALINTEGRATION_H

#include <common/Common_Deffinitions.h>

/// \brief First Order Digital Integration IIR Filter
class FirstOrderDigitalIntegration
{
public:
    /// \brief Constructor
    /// \param sampleRate Samples per second required for integration.
    FirstOrderDigitalIntegration(RADIO_DATA_TYPE sampleRate);

    /// \brief integrate data array
    /// \param array array of input/output data to integrate
    /// \param length length of the input array
    void integrate(RADIO_DATA_TYPE *array, size_t length);

private:
    RADIO_DATA_TYPE tStep; ///< Internal timestep based on constructor samplerate
};


#endif //RADIONODE_FIRSTORDERDIGITALINTEGRATION_H
