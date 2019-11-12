//
// Created by Micheal Cowan on 2019-08-02.
//

#include <streams/RawDataOutput.h>
#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/FFT.h>

RawDataOutput::RawDataOutput(SafeBufferPoolQueue* BBP_Buffer_Pool,
                       std::string name,
                       std::string address,
                       std::string m_td_port,
                       std::string m_fd_port,
                       size_t array_size)
: Baseband_Stream([this](BBP_Block* block)
                    {
                        // Decimate the  Raw input down to the 16000 that is expected
                        decimate(block, 64);
                        // order matters here
                        write_to_file(block);
                        publish_TD(block);
                        publish_FD(block);
                        return 0;
                    },
                    BBP_Buffer_Pool)
, m_file(std::string(name + ".pcm"), std::ios::binary | std::ios::trunc | std::ios::out)
, m_TD_Blaster(std::string(name + "_TD_Blaster"), address, m_td_port)
, m_FD_Blaster(std::string(name + "_FD_Blaster"), address, m_fd_port)
, p_real_array(new RADIO_DATA_TYPE[array_size])
, m_array_size(array_size)
{
}

RawDataOutput::~RawDataOutput()
{
    delete[] p_real_array;
}

void RawDataOutput::write_to_file(BBP_Block* block)
{
    block_to_arrays(block);
    // Write full array_size 16000 samples / second to file
    m_file.write((const char*)p_real_array, m_array_size*sizeof(RADIO_DATA_TYPE));
}

void RawDataOutput::publish_TD(BBP_Block* block)
{
    // Decimate 16000*8 down to a sendable amount
    decimate(block, 32);
    m_TD_Blaster.send((const char*)&block->points[0], block->n_points*sizeof(Complex));
}

void RawDataOutput::publish_FD(BBP_Block* block)
{
    // Generate FFT
    fft(block->points, block->number_of_points()); // Block has already been decimated by previous call
    block_to_arrays(block);
    m_FD_Blaster.send((const char*)p_real_array, block->n_points*sizeof(RADIO_DATA_TYPE));
}

void RawDataOutput::block_to_arrays(BBP_Block* block)
{
    for(size_t index = 0; index < m_array_size; index++)
    {
        p_real_array[index] = block->points[index].real();
    }
}
