/// @file SDR/RadioNode/src/libraries/DigitalSignalProcessing/NotchFilter.h


#ifndef RADIONODE_NOTCHFILTER_H
#define RADIONODE_NOTCHFILTER_H

#include <stdint.h>
#include <math.h>
#include <common/Common_Deffinitions.h>

/// \brief A simple Notch Filter IIR
class NotchFilter
{
public:
    /// \brief Constructor
    /// \param notch_bw Bandwidth of the notch in hertz
    /// \param center_freq Center frequency in hertz
    NotchFilter(RADIO_DATA_TYPE notch_bw, RADIO_DATA_TYPE center_freq);

    /// \brief Filter an array of data poitns
    /// \param block_in Input/Output array of data
    /// \param length Length of the array
    void filter(RADIO_DATA_TYPE* block_in, size_t length);

private:
    RADIO_DATA_TYPE R;    ///< R Value
    RADIO_DATA_TYPE K;    ///< K Value
    RADIO_DATA_TYPE a0;   ///< a0 coefficient
    RADIO_DATA_TYPE a1;   ///< a1 coefficient
    RADIO_DATA_TYPE a2;   ///< a2 coefficient
    RADIO_DATA_TYPE b1;   ///< b1 coefficient
    RADIO_DATA_TYPE b2;   ///< b2 coefficient
};


#endif //RADIONODE_NOTCHFILTER_H
