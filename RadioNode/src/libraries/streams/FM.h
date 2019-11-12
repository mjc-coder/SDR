//
// Created by Micheal Cowan on 9/27/19.
//

#ifndef RADIONODE_FM_H
#define RADIONODE_FM_H

#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/LowPassFilter.h>
#include <FIR-filter-class/filt.h>
#include <iostream>

template<class data_type>
class FM
{
public:

    FM(double tx_sample_rate, double rx_sample_rate, double MarkFreq=10, double SpaceFreq=15)
    : m_tx_sample_rate(tx_sample_rate)
    , m_rx_sample_rate(rx_sample_rate)
    , MARK_FREQ(MarkFreq)
    , SPACE_FREQ(SpaceFreq)
    , m_alpha(0)
    , m_real(0)
    , m_imag(0)
    {
    }

    ~FM()
    {
    }

    void modulate(uint8_t* data_in, size_t length, data_type* modulated_real, data_type* modulated_imag, size_t length_modulated, size_t samples_per_bit)
    {
        double freq = 0;
        const int upscale = length_modulated/length;
        const double delta_t = 1.0/m_tx_sample_rate;
        const double RootRaised = sqrt(2.0)/static_cast<double>(samples_per_bit);

        for(size_t i = 0; i < length; i++)
        {
            // compute appropriate frequency
            if(data_in[i] == 0)
            {
                freq = SPACE_FREQ*2.0*PI;
            }
            else
            {
                freq = MARK_FREQ*2.0*PI;
            }
            std::cout << "Upscale " << upscale  << "   " << freq << std::endl;

            // generate output tone
            for (size_t j = 0; j < upscale; j++)
            {
                // compute complex output
                modulated_real[i*upscale+j] = RootRaised*cos(freq*delta_t*j);
                modulated_imag[i*upscale+j] = 0;
            }
        }
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


    size_t demodulate(data_type* real, data_type* imag, uint8_t* output, size_t number_of_points, size_t downsample)
    {
        size_t length = number_of_points/downsample;

        data_type* WAVE1 = new RADIO_DATA_TYPE[downsample];
        data_type* WAVE2 = new RADIO_DATA_TYPE[downsample];
        data_type* op1 = new RADIO_DATA_TYPE[number_of_points+2*downsample];
        data_type* op2 = new RADIO_DATA_TYPE[number_of_points+2*downsample];

        m_alpha = ((1.0/(2000000.0))/(1.0/(5000.0)));

        for(int i = 0; i < number_of_points; i++)
        {
            m_real = m_alpha * (real[i] - m_real);
            m_imag = m_alpha * (imag[i] - m_imag);
            real[i] = m_real;
        }

        std::cout << "Demodulating FM" << std::endl;

        const double delta_t = 1.0/m_rx_sample_rate;
        // Generate Mark and Space convolution waves
        for(size_t i = 0; i < downsample; i++)
        {
            WAVE1[i] = sqrt(2.0/downsample)*cos(2.0*PI*MARK_FREQ*i*delta_t);
            WAVE2[i] = sqrt(2.0/downsample)*cos(2.0*PI*SPACE_FREQ*i*delta_t);
        }

        // convolution 1 is input with WAVE 1
        convolve(real, number_of_points, WAVE1, downsample, op1, number_of_points);
        convolve(real, number_of_points, WAVE2, downsample, op2, number_of_points);


//        op1 = conv(y, sqrt(2/T)*cos(2*pi*10*t)); % correlating with frequency 1 -- MARK
//        op2 = conv(y, sqrt(2/T)*cos(2*pi*20*t)); % correlating with frequency 2 -- SPACE
//                                                                              % demodulation
//        ipHat = [real(op1(T+1:T:end)) < real(op2(T+1:T:end))];


 //       std::cout << "DS " << downsample << "  " << number_of_points << "  " << length << std::endl;
        for(size_t i = downsample-1, j = 0; i < number_of_points; i+=downsample, j++)
        {
            output[j] = (abs(op1[i]) > abs(op2[i])) ? 1:0;
//            std::cout << (int)output[j] << "  " << abs(op1[i]) << "  " << abs(op2[i]) << std::endl;
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
};


#endif //RADIONODE_FM_H

