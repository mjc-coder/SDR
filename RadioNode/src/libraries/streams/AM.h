//
// Created by Micheal Cowan on 9/27/19.
//

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

template<class data_type>
class AM
{
public:
    static constexpr double THRESHOLD = 80; //sqrt(125.0*125.0 + 125.0*125.0) / 2;

public:
    AM()
    {
    }

    ~AM()
    {
    }

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
    size_t m_array_size;
    data_type* m_RealData;
    data_type m_max_normalization;
    data_type m_real;
    data_type m_imag;
    data_type m_alpha;
    data_type m_alpha2;
    data_type m_output_temp;
};


#endif //RADIONODE_AM_H

