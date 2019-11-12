//
// Created by Micheal Cowan on 2019-08-01.
//

#ifndef RADIONODE_AM_DEMODULATOR_H
#define RADIONODE_AM_DEMODULATOR_H

#include <string>
#include <common/SafeBufferPoolQueue.h>
#include <streams/DataOutput.h>
#include <FIR-filter-class/filt.h>


class AM_Demodulator : public Baseband_Stream {
public:
    AM_Demodulator(SafeBufferPoolQueue* BBP_Buffer_Pool,
                   std::string name,
                   std::string address,
                   std::string m_td_port,
                   std::string m_fd_port,
                   size_t array_size);

    ~AM_Demodulator();

    void demod(BBP_Block* block);

private:
    DataOutput m_dataoutput;
    size_t m_array_size;
    RADIO_DATA_TYPE* m_pcm;
    RADIO_DATA_TYPE m_max_normalization;
    Filter filter;
};


#endif //RADIONODE_AM_DEMODULATOR_H
