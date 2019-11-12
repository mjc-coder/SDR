//
// Created by Micheal Cowan on 2019-08-02.
//

#include <streams/DataOutput.h>
#include <network/UDP_Blaster.h>
#include <DigitalSignalProcessing/Resample.h>
#include <DigitalSignalProcessing/FFT.h>

DataOutput::DataOutput(SafeBufferPoolQueue* BBP_Buffer_Pool,
                       std::string name,
                       std::string address,
                       std::string m_td_port,
                       std::string m_fd_port,
                       size_t array_size)
: Baseband_Stream([this](BBP_Block* block)
                    {
                        // order matters here
                        write_to_file(block);
                        // decimates 32000 by 16
                        publish_TD(block);
                        // already decimated
                        publish_FD(block);
                        return 0;
                    },
                    BBP_Buffer_Pool)
, m_file(std::string(name + ".pcm"), std::ios::binary | std::ios::trunc | std::ios::out)
, m_TD_Blaster(std::string(name + "_TD_Blaster"), address, m_td_port, "0")
, m_FD_Blaster(std::string(name + "_FD_Blaster"), address, m_fd_port, "0")
, p_real_array(new RADIO_DATA_TYPE[array_size])
, m_array_size(array_size)
{
}

DataOutput::~DataOutput()
{
    delete[] p_real_array;
}

void DataOutput::write_to_file(BBP_Block* block)
{
    block_to_arrays(block);
    // Write full array_size 32000 samples / second to file
    m_file.write(reinterpret_cast<const char*>(&p_real_array[1]), (sizeof(RADIO_DATA_TYPE) * (m_array_size-2)));
}

void DataOutput::publish_TD(BBP_Block* block)
{
    // Decimate 16000 down to a sendable amount
    decimate(block, 32);
    m_TD_Blaster.send((const char*)&block->points[0], block->n_points*sizeof(Complex));
}

void DataOutput::publish_FD(BBP_Block* block)
{
    // Generate FFT
    fft(block->points, block->number_of_points()); // Block has already been decimated by previous call
    block_to_arrays(block);
    m_FD_Blaster.send((const char*)p_real_array, block->n_points*sizeof(RADIO_DATA_TYPE));
}

void DataOutput::block_to_arrays(BBP_Block* block)
{
    memset(p_real_array, 0, m_array_size*sizeof(RADIO_DATA_TYPE));
    for(size_t index = 0; index < m_array_size; index++)
    {
        p_real_array[index] = block->points[index].real();
    }
}

