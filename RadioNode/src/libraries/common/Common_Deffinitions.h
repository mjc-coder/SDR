/// @file SDR/RadioNode/src/libraries/common/Common_Deffinitions.h

#ifndef RADIONODE_COMMON_DEFFINITIONS_H
#define RADIONODE_COMMON_DEFFINITIONS_H

#include <stdlib.h>
#include <cmath>
#include <complex>
#include <valarray>

typedef double RADIO_DATA_TYPE;                                             ///< Common Data type
typedef std::complex<RADIO_DATA_TYPE> Complex;                              ///< Alias for Complex data type
typedef std::valarray<Complex> Complex_Array;                               ///< Alias for Complex Array of values
const RADIO_DATA_TYPE PI = 3.141592653589793238460;                         ///< Constant Definition for PI

/// \brief Macro for converting Mega-hz to hz
#define MHZ_TO_HZ(freq) (freq * 1000000)

/// \brief Block Read Size -- 65536 Samples / 32768 Points / 131072 bytes
#define BLOCK_READ_SIZE 240000/2

/// \brief Hardware decoding type
enum HardwareType
{
    CPU,    ///< Central Processing Unit
    GPU     ///< Graphical Processing Unit
};

/// \brief Auto or Manual AGC Gain Mode.
/// \details Most usage is in fully manual mode.
enum GainMode
{
    Auto,
    Manual
};

#endif //RADIONODE_COMMON_DEFFINITIONS_H
