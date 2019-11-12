//
// Created by Micheal Cowan on 2019-07-14.
//

#include "Scrap.h"
#include <math.h>
#include <complex>
#include <liquid/liquid.h>
#include <fstream>





// reference: https://www.embedded.com/design/configurable-systems/4212086/DSP-Tricks--Frequency-demodulation-algorithms-#
void fm_demod(BBP_Block* block, std::ofstream& f, Complex* p0, Complex* p1)
{
    float ang[BLOCK_READ_SIZE];
    int num_points = 0;

    // Downsample by 4
    fifth_order filter;
    filter.decimate(block, 4);

    // Point 0
    float quad = block->points[0].imag();
    float quad_delay_1 = p1->imag();
    float quad_delay_2 = p0->imag();
    float inphase = block->points[0].real();
    float inphase_delay_1 = p1->real();
    float inphase_delay_2 = p0->real();

    float quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    float imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    ang[0] = quad_prime - imag_prime;

    // Point 1
    quad = block->points[1].imag();
    quad_delay_1 = block->points[0].imag();
    quad_delay_2 = p1->imag();
    inphase = block->points[1].real();
    inphase_delay_1 = block->points[0].real();
    inphase_delay_2 = p1->real();

    quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    ang[1] = quad_prime - imag_prime;

    // Loop through remaining
    for(size_t index = 2; index < block->number_of_points(); index++)
    {
        quad = block->points[index].imag();
        quad_delay_1 = block->points[index-1].imag();
        quad_delay_2 = block->points[index-2].imag();
        inphase = block->points[index].real();
        inphase_delay_1 = block->points[index-1].real();
        inphase_delay_2 = block->points[index-2].real();

        quad_prime = (quad - quad_delay_2) * inphase_delay_1;
        imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

        ang[index] = quad_prime - imag_prime;
    }

    // Store history
    p0->real(block->points[block->number_of_points()-2].real());
    p0->imag(block->points[block->number_of_points()-2].imag());
    p1->real(block->points[block->number_of_points()-1].real());
    p1->imag(block->points[block->number_of_points()-1].imag());

    NotchFilter notch(0.0033, 64500000);
    notch.filter(ang, block->number_of_points());

    LowPassFilter lpf2(20000, 1.0/512000.0);
    for(size_t i = 0; i < block->number_of_points(); i++)
    {
        ang[i] = lpf2.update(ang[i]);
    }

    num_points = filter.decimate(ang, block->number_of_points(), 16);

    normalize(ang, num_points);

    f.write(reinterpret_cast<const char*>(&ang[1]), sizeof(float) * (num_points-2));
}

void am_demod(BBP_Block* block, std::ofstream& file, LowPassFilter* lpf)
{
    float pcm[block->number_of_points()];
    size_t num_points = 0;

    for (size_t i = 0; i < block->number_of_points(); i++)  // Decimate to 48000 hz
    {
        // hypot uses floats but won't overflow
        //r[i/2] = (int16_t)hypot(lp[i], lp[i+1]);
        pcm[i] = lpf->update(sqrt( ((float)block->points[i].real() * (float)block->points[i].real()) + ((float)block->points[i].imag() * (float)block->points[i].imag()))); // ABS
    }

    num_points = decimate(pcm, block->number_of_points(), 64); // 5 for test file

    normalize(pcm, num_points);

    file.write(reinterpret_cast<const char*>(&pcm[1]), (sizeof(float) * (num_points-2)));
}



void FreqModulate(float* inputData, size_t bufferSize, float* real, float* imag, float sampleRate)
{
    FILE *fFMData ;
    size_t nIndex = 0;
    double nCarrierFreq = 10000;
    double nAmplitude = 10000;
    double fBeta = 0.9;
    double fCarrierSignal = 10000;
    double fModulatedSignal = 0;
    double fCarrier = 0;
    double fSine = 0;
    double fModulation = 0;
//_dataSize = 11025;
    float fTimeDiv = (float)(1.0f / sampleRate) ;

    fFMData = fopen("D:\\FMData.txt", "w") ;
    while(nIndex < bufferSize)
    {
        fCarrierSignal = 2 * PI * nCarrierFreq * (nIndex * fTimeDiv) ;

        fModulatedSignal = 2 * PI * inputData[nIndex] * (nIndex * fTimeDiv) ;

        //fCarrier = cos(fCarrierSignal) ;

        fSine = sin(fCarrierSignal) ;

        fModulation = sin(fModulatedSignal);

        //m_fSample[nIndex] = nAmplitude + (nAmplitude * fCarrier) - (fBeta * nAmplitude * fSine * fModulation) ;
     //   m_fSample[nIndex] = nAmplitude + cos(fCarrierSignal + fBeta * fModulation) ;

     //   fprintf(fFMData, "%f\n", m_fSample[nIndex]) ;

        nIndex++;
    }
}


void frequency_modulation(size_t array_length,
                          float* input_items,
                          float* real_output_items,
                          float* imag_output_items,
                          float sample_rate)
{
    // value needs to be between [0,1]
   float K = 2.0*PI*(5000.0)/sample_rate;  // Sensitivity parameter  5Khz - Narrow band
                                            //                       75Khz - Wideband
    float Fc = 0;
    std::complex<float> J(0,1);
    float cummulative_message = 0;
    float d_phase = 0;

    for (size_t i = 0; i < array_length; i++)
    {
        cummulative_message+=input_items[i];
        std::cout << cummulative_message << std::endl;

        //d_phase = cos(2.0*PI*Fc*(i/sample_rate) + K*cummulative_message - PI/2);
        real_output_items[i] = cos(2.0*PI*Fc*(i/sample_rate) + K*cummulative_message - 0);
        imag_output_items[i] = sin(2.0*PI*Fc*(i/sample_rate) + K*cummulative_message - 0);
    }
}

void freqModulate(size_t array_size,
                  float* input_items,
                  float* real_output_items,
                  float* imag_output_items,
                  size_t samplerate)
{
    // value needs to be between [0,1]
    //float K = 2.0*PI*(75000.0)/(float)samplerate;  // Sensitivity parameter  5Khz - Narrow band
    //                       75Khz - Wideband

    float K = 10000;

    FirstOrderDigitalIntegration integrator(samplerate);

    for(size_t i = 0; i < array_size; i++)
    {
        input_items[i] *= K;
    }

    integrator.integrate(input_items, array_size);

    for(int i = 0; i < array_size; i++)
    {
        Complex p = std::polar(1.0f, input_items[i]);
        real_output_items[i] = p.real() + real_output_items[i];
        imag_output_items[i] = p.imag() + imag_output_items[i];
        std::cout << p.real() << "  " << p.imag() << std::endl;
    }
}


void liquidFreqModulate(size_t array_size,
                        float* input_items,
                        float* real_output_items,
                        float* imag_output_items,
                        size_t samplerate)
{
    float K = 0.3;      // f_c / f_sample_rate
    std::cout << "K Term: " << K << std::endl;
    freqmod fmod = freqmod_create(K);

    liquid_float_complex* s = new liquid_float_complex[array_size];
    std::ofstream fout("fm.dump", std::ios::binary);


    freqmod_modulate_block(fmod,input_items,array_size,s);

    for(size_t i = 0; i < array_size; i++)
    {
        real_output_items[i] = s[i].real();
        imag_output_items[i] = s[i].imag();
        float f1 = s[i].real();
        float f2 = s[i].imag();
        fout.write((char*)&f1, sizeof(float));
        fout.write((char*)&f2, sizeof(float));
    }

    freqmod_destroy(fmod);
    delete[] s;
    fout.close();
}

