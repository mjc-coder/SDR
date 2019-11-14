//
// Created by Micheal Cowan on 9/27/19.
//

#ifndef RADIONODE_FM_H
#define RADIONODE_FM_H

#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/LowPassFilter.h>
#include <FIR-filter-class/filt.h>
#include <iostream>
#include <fstream>

template<class data_type>
class FM
{
public:

    FM(double tx_sample_rate, double rx_sample_rate, double MarkFreq=19000, double SpaceFreq=20000)
    : m_tx_sample_rate(tx_sample_rate)
    , m_rx_sample_rate(rx_sample_rate)
    , MARK_FREQ(MarkFreq)
    , SPACE_FREQ(SpaceFreq)
    , m_alpha(0)
    , m_real(0)
    , m_imag(0)
    , fout("low.dat", std::ios::out | std::ios::trunc)
    , fout2("high.dat", std::ios::out | std::ios::trunc)
    {
    }

    ~FM()
    {
    }

    void modulate(uint8_t* data_in, size_t length, data_type* modulated_real, data_type* modulated_imag, size_t length_modulated, size_t samples_per_bit)
    {
        const double delta_t = 1.0/m_tx_sample_rate;
        const double RootRaised = sqrt(2)/sqrt(samples_per_bit);

        data_type* Wave1 = new data_type[samples_per_bit];
        data_type* Wave2 = new data_type[samples_per_bit];

        // Build Waves
        for(size_t i = 0; i < samples_per_bit; i++)
        {
            Wave1[i] = RootRaised*cos(2.0*PI*MARK_FREQ*delta_t*i);
            Wave2[i] = RootRaised*cos(2.0*PI*SPACE_FREQ*delta_t*i);
        }

        for(size_t i = 0; i < length; i++)
        {
            // compute appropriate frequency
            if(data_in[i] == 0)
            {
                memcpy(&modulated_real[i*samples_per_bit], Wave2, samples_per_bit*sizeof(data_type));
            }
            else
            {
                memcpy(&modulated_real[i*samples_per_bit], Wave1, samples_per_bit*sizeof(data_type));
            }
        }

        delete[] Wave1;
        delete[] Wave2;
    }

    // INPUT 1 - Read Data
    // INPUT 2 - Pulse SHAPE
    void convolve(const data_type* input1, size_t length1, const data_type* input2, size_t length2, data_type* output, size_t lengthOutput)
    {
        RADIO_DATA_TYPE* real_data = new RADIO_DATA_TYPE[length1+(length2*2)];  // Upsample 8, and include forward buffer for convolution

        memset(real_data, 0, (length1+(length2*2))*sizeof(RADIO_DATA_TYPE));

        // Initialize data
        for(size_t i = 0; i < length1; i++)
        {
            real_data[i+length2] = input1[i];
        }

        for(size_t i = 0; i < length1; i++) {
            for (size_t j = 0; j < length2; j++) {
                output[i] += real_data[i + length2 - j] * input2[length2 - j - 1];
            }
        }

        delete[] real_data;
    }


    size_t demodulate(data_type* real, data_type* imag, uint8_t* output, size_t number_of_points, size_t ds)
    {
        size_t length = number_of_points/ds;

        data_type* WAVE1 = new RADIO_DATA_TYPE[ds];
        data_type* WAVE2 = new RADIO_DATA_TYPE[ds];
        data_type* op1 = new RADIO_DATA_TYPE[number_of_points+2*ds];
        data_type* op2 = new RADIO_DATA_TYPE[number_of_points+2*ds];

        const double delta_t = 1.0/m_rx_sample_rate;
        // Generate Mark and Space convolution waves
        for(size_t i = 0; i < ds; i++)
        {
            WAVE1[i] = sqrt(2.0/ds)*cos(2.0*PI*MARK_FREQ*i*delta_t);
            WAVE2[i] = sqrt(2.0/ds)*cos(2.0*PI*SPACE_FREQ*i*delta_t);
        }

        // convolution 1 is input with WAVE 1
        convolve(real, number_of_points, WAVE1, ds, op1, number_of_points);
        convolve(real, number_of_points, WAVE2, ds, op2, number_of_points);


        for(size_t i = ds-1, j = 0; i < number_of_points; i+=ds, j++)
        {
            output[j] = (op1[i] > op2[i]) ? 1:0;
        }

        delete[] WAVE1;
        delete[] WAVE2;
        delete[] op1;
        delete[] op2;

        return length;
    }


private:
    data_type* m_RealData;
    double m_tx_sample_rate;
    double m_rx_sample_rate;
    double MARK_FREQ;
    double SPACE_FREQ;
    double m_alpha;
    double m_real;
    double m_imag;
    std::ofstream fout;
    std::ofstream fout2;
};


#endif //RADIONODE_FM_H

