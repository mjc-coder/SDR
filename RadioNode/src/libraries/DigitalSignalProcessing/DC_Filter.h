/// @file SDR/RadioNode/src/libraries/DigitalSignalProcessing/DC_Filter.h


#ifndef RADIONODE_DC_FILTER_H
#define RADIONODE_DC_FILTER_H

#include <common/Common_Deffinitions.h>

/// \brief A DC noise filter to remove a DC signal or ultra low frequency signal
// Original reference found here: https://www.dsprelated.com/freebooks/filters/DC_Blocker.html
class DC_Filter
{
public:
    /// \brief constructor
    /// \param R must be between 0.0 and 1.0; and 0.995 for 44.1khz as a reference
    DC_Filter(RADIO_DATA_TYPE R = 0.995);

    /// \brief IIR filter for an array of data
    /// \param input Input array of data
    /// \param output Output filtered data
    /// \param array_size Number of points in the two arrays
    void update(RADIO_DATA_TYPE *input, RADIO_DATA_TYPE *output, size_t array_size);

private:
    RADIO_DATA_TYPE m_R; /// R value provided initially.
};


#endif //RADIONODE_DC_FILTER_H
