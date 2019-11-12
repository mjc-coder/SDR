//
// Created by Micheal Cowan on 2019-07-14.
//

#ifndef RADIONODE_SCRAP_H
#define RADIONODE_SCRAP_H


#include "common/Baseband_Stream.h"
#include <fstream>
#include <stdint.h>

void fm_demod(BBP_Block* block, std::ofstream& file, Complex* p0, Complex* p1);

void am_demod(BBP_Block* block, std::ofstream& file, LowPassFilter* lpf);

void FreqModulate(float* inputData, size_t bufferSize, float* real, float* imag, float sampleRate);

void frequency_modulation(size_t array_size,
                          float* input_items,
                          float* real_output_items,
                          float* imag_output_items,
                          float samplerate);

void freqModulate(size_t array_size,
                    float* input_items,
                    float* real_output_items,
                    float* imag_output_items,
                    size_t samplerate);

void liquidFreqModulate(size_t array_size,
                        float* input_items,
                        float* real_output_items,
                        float* imag_output_items,
                        size_t samplerate);

#endif //RADIONODE_SCRAP_H
