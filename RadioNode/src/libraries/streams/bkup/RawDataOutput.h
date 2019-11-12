//
// Created by Micheal Cowan on 2019-08-02.
//

#ifndef RADIONODE_RAWDATAOUTPUT_H
#define RADIONODE_RAWDATAOUTPUT_H

#include <common/SafeBufferPoolQueue.h>
#include <common/Baseband_Stream.h>
#include <fstream>
#include <network/UDP_Blaster.h>

class RawDataOutput : public Baseband_Stream
{
public:
    RawDataOutput( SafeBufferPoolQueue* BBP_Buffer_Pool,
                std::string name,
                std::string address,
                std::string m_td_port,
                std::string m_fd_port,
                size_t array_size);

    ~RawDataOutput();

private:
    void write_to_file(BBP_Block* block);
    void publish_TD(BBP_Block* block);
    void publish_FD(BBP_Block* block);

private:
    std::ofstream m_file;
    UDP_Blaster m_TD_Blaster;
    UDP_Blaster m_FD_Blaster;
    void block_to_arrays(BBP_Block* block);
    RADIO_DATA_TYPE* p_real_array;
    size_t m_array_size;
};


#endif //RADIONODE_DATAOUTPUT_H
