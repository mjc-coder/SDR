/// @file SDR/RadioNode/src/libraries/streams/AM.h

#ifndef RADIONODE_AM_H
#define RADIONODE_AM_H

#include <string>
#include <common/SafeBufferPoolQueue.h>
#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/Normalize.h>
#include <iostream>
#include <chrono>
#include <DigitalSignalProcessing/LowPassFilter.h>

using namespace std::chrono;

/// \brief Binary AM Modulation and Demodulation Interface
/// \tparam data_type Point data types, usually double or uint8
template<class data_type>
class AM
{
public:
    /// Threshold for decision making for ones or zeros
    static constexpr double THRESHOLD = 80; //sqrt(125.0*125.0 + 125.0*125.0) / 2;

public:
    /// Constructor
    AM()
    {
    }

    /// Destructor
    ~AM()
    {
    }

    /// \brief AM Modulation routine, this function will upsample the input signal to fit the output.
    /// \param data_in Input data stream of 1's or 0's
    /// \param length number of input data points
    /// \param modulated_real output data stream, all real data.
    /// \param length_modulated total number of points in the modulated data stream.
    void modulate(uint8_t* data_in, size_t length, data_type* modulated_real, data_type* /*modulated_imag*/, size_t length_modulated)
    {
        m_RealData = new data_type[length];

        for(size_t i = 0; i < length; i++)
        {
            m_RealData[i] = data_in[i];
        }

        // all data will be 1's or 0's
        (void)raw_upsample<data_type>(m_RealData, length, modulated_real, length_modulated, static_cast<double>(length_modulated)/static_cast<double>(length));

        delete[] m_RealData;
    }

    /// \brief Demodulation routine
    /// \param real input stream of real data.
    /// \param imag input stream of imaginary data.
    /// \param output Stream of output data in binary
    /// \param number_of_points in the output stream.
    /// \param downsample factor to downsample the input to the output.
    /// \return Total number of bits after demodulation.
    size_t demodulate(data_type* real, data_type* imag, uint8_t* output, size_t number_of_points, size_t downsample)
    {        
         size_t out_index = 0;

        // Complex Absolute -- downsamples at the same time
        for (size_t i = 0; i < number_of_points; i+=downsample, out_index++)
        {
            //std::cout << m_output_temp << std::endl;
            // hypot uses floats but won't overflow
            // data = i^2 + q^2
            output[out_index] = ((sqrt( real[i] * real[i] + imag[i] * imag[i])  >= THRESHOLD) ? 1.0 : 0.0);
        }

        return out_index;
    }

private:
    data_type* m_RealData;  ///< Real Data stream used for modulation.
};


#endif //RADIONODE_AM_H

