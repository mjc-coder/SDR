//
// Created by Micheal Cowan on 2019-08-01.
//

#ifndef RADIONODE_FM_DEMODULATOR_H
#define RADIONODE_FM_DEMODULATOR_H

#include <string>
#include <streams/DataOutput.h>
#include <common/Common_Deffinitions.h>
#include <DigitalSignalProcessing/LowPassFilter.h>
#include <DigitalSignalProcessing/NotchFilter.h>

class FM_Demodulator : public Baseband_Stream {
public:
    FM_Demodulator( SafeBufferPoolQueue* BBP_Buffer_Pool,
                    std::string name,
                    std::string address,
                    std::string m_td_port,
                    std::string m_fd_port,
                    size_t array_size);

    ~FM_Demodulator();

    void demod(BBP_Block* block);

    void demod2(BBP_Block* block);

private:
    DataOutput m_dataoutput;
    RADIO_DATA_TYPE m_ang[1024000];
    RADIO_DATA_TYPE quad;
    RADIO_DATA_TYPE quad_delay_1;
    RADIO_DATA_TYPE quad_delay_2;
    RADIO_DATA_TYPE inphase;
    RADIO_DATA_TYPE inphase_delay_1;
    RADIO_DATA_TYPE inphase_delay_2;
    RADIO_DATA_TYPE quad_prime;
    RADIO_DATA_TYPE imag_prime;
    Complex p0;
    Complex p1;
    LowPassFilter m_lpf;
    NotchFilter m_notch;
    RADIO_DATA_TYPE normalization_max;
};

#endif //RADIONODE_FM_DEMODULATOR_H
