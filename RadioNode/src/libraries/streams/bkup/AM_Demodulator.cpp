//
// Created by Micheal Cowan on 2019-08-01.
//

#include <streams/AM_Demodulator.h>
#include <FIR-filter-class/filt.h>
#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/Normalize.h>

AM_Demodulator::AM_Demodulator(SafeBufferPoolQueue* BBP_Buffer_Pool,
                               std::string name,
                               std::string address,
                               std::string m_td_port,
                               std::string m_fd_port,
                               size_t array_size)
: Baseband_Stream([this](BBP_Block* block)
                  {
                      demod(block);
                  }, BBP_Buffer_Pool)
, m_dataoutput(BBP_Buffer_Pool, name, address, m_td_port, m_fd_port, array_size)
, m_array_size(array_size)
, m_pcm(new RADIO_DATA_TYPE[array_size])
, m_max_normalization(0)
, filter(BPF, 51, 240.0, 0.01,20.0)
{
    memset(m_pcm, 0, m_array_size);
    this->add_next_buffer(&m_dataoutput);
}

AM_Demodulator::~AM_Demodulator()
{
    delete[] m_pcm;
}

void AM_Demodulator::demod(BBP_Block* block)
{
    size_t num_points = 0;

    // Complex Absolute
    for (size_t i = 0; i < block->number_of_points(); i++)
    {
        // hypot uses floats but won't overflow
        //r[i/2] = (int16_t)hypot(lp[i], lp[i+1]);
        m_pcm[i] = filter.do_sample(sqrt( (block->points[i].real() * block->points[i].real()) + (block->points[i].imag() * block->points[i].imag())));
    }

    num_points = decimate(m_pcm, block->number_of_points(), 5); // decimation of 5

    // Run an average over this thing to remove squelches or spikes
    for(int i = 5; i < num_points; i++)
    {
        m_pcm[i] = (m_pcm[i] + m_pcm[i-1] + m_pcm[i-2] + m_pcm[i-3] + m_pcm[i-4])/5.0;
    }

    m_max_normalization = normalize(m_pcm, num_points, m_max_normalization);

    // write normalized points back to block array
    for(size_t i = 0; i < num_points; i++)
    {
        block->points[i].real(m_pcm[i]);
        block->points[i].imag(0);
    }
    block->n_points = num_points;
}