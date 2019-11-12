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

using namespace std::chrono;

template<class data_type>
class AM
{
public:
    AM()
    {
    }


    AM(SafeBufferPoolQueue* /*BBP_Buffer_Pool*/,
            std::string /*name*/,
            std::string /*address*/,
            std::string /*m_td_port*/,
            std::string /*m_fd_port*/,
            size_t /*array_size*/)
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


    void demodulate(BBP_Block* input, BBP_Block* output)
    {
        m_RealData = new data_type[input->number_of_points()];
        // Complex Absolute
        for (size_t i = 0; i < input->number_of_points(); i++)
        {
            // hypot uses floats but won't overflow
            // data = i^2 + q^2
            m_RealData[i] = sqrt( (input->points[i].real() * input->points[i].real()) + (input->points[i].imag() * input->points[i].imag()));
        }

        size_t num_points = decimate(m_RealData, input->number_of_points(), 8); // decimation of 8

        m_max_normalization = normalize(m_RealData, num_points, m_max_normalization); // normalize data to 1.

        // write normalized points back to block array
        for(size_t i = 0; i < num_points; i++)
        {
            output->points[i].real((m_RealData[i] >= 0.5) ? 1.0 : 0.0); // decision tree
            output->points[i].imag(0);
        }

        output->n_points = num_points;
        delete[] m_RealData;
    }

    size_t demodulate(Complex_Array& input, Complex_Array& output, size_t number_of_points, size_t downsample)
    {
        m_RealData = new data_type[number_of_points];
        memset(m_RealData, 0, sizeof(data_type)*number_of_points);

        // Complex Absolute
        for (size_t i = 0; i < number_of_points; i++)
        {
            // hypot uses floats but won't overflow
            // data = i^2 + q^2
            m_RealData[i] = sqrt( abs(input[i].real() * input[i].real()) + abs(input[i].imag() * input[i].imag()) );
        }
        size_t num_points = decimate(m_RealData, number_of_points, downsample); // decimation of 8

        // write normalized points back to block array
        for(size_t i = 0; i < num_points; i++)
        {
            output[i].real((m_RealData[i] >= 63.5) ? 1.0 : 0.0); // decision tree
            output[i].imag(0);
        }

        delete[] m_RealData;
        return num_points;
    }

    size_t demodulate(Complex_Array& input, uint8_t* output, size_t number_of_points, size_t downsample)
    {
        // Complex Absolute
        for (size_t i = 0; i < number_of_points; i++)
        {
            // hypot uses floats but won't overflow
            // data = i^2 + q^2
            output[i] = (sqrt( abs(input[i].real() * input[i].real()) + abs(input[i].imag() * input[i].imag()) ) >= 63.5) ? 1.0 : 0.0;
        }

        return decimate(output, number_of_points, downsample); // decimation of 8
    }
private:
    size_t m_array_size;
    data_type* m_RealData;
    data_type m_max_normalization;
};


#endif //RADIONODE_AM_H

