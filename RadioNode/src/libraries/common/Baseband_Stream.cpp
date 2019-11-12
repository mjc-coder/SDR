

#include <common/Baseband_Stream.h>
#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <common/SafeBufferPoolQueue.h>
#include <iostream>

const Complex NULLPOINT = Complex(0,0);

Baseband_Stream::Baseband_Stream(std::function<void(BBP_Block*)> post_cb,
                                 SafeBufferPoolQueue* BBP_Buffer_Pool)
: m_buffer(0)
, m_post_cb(post_cb)
, m_BBP_Buffer_Pool(BBP_Buffer_Pool)
, m_future_obj(m_exit_signal.get_future())
, process_thread(std::thread([this]()
     {
         while(this->m_future_obj.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout)
         {
             if(m_buffer.get_available_blocks() > 0)
             {
                // Pop
                 BBP_Block* block = m_buffer.next();

                // Post CB
                 m_post_cb(block);

                // If the number of buffers is greater than 1, duplicate and push
                // buffers to the appropriate places
                if(m_next_transform.size() > 1)
                {
                    // Copy and Push to multiple locations
                    for(size_t i = 1; i < m_next_transform.size(); i++)
                    {
                        // Grab a frame from the buffer pool
                        BBP_Block* block_copy = m_BBP_Buffer_Pool->next(); // thread safe

                        if(block_copy != nullptr)
                        {
                            // Hard copy the memory
                            block_copy->hard_copy(*block);

                            // Push to next
                            m_next_transform[i]->load_buffer(block_copy);
                        }
                        else
                        {
                            std::cout << "Dude were losing data... ran out of frames..." << std::endl;
                        }
                    }
                    m_next_transform[0]->load_buffer(block); // Push current block to the first
                }
                else if(m_next_transform.size() == 1) // Only one downstream, just move the pointer to the next stage
                {
                    m_next_transform[0]->load_buffer(block);
                }
                else
                {
                    // No downstream... thats a problem unless its the end of the pipe
                    // So we will return them to the buffer pool
                    block->reset();
                    m_BBP_Buffer_Pool->push(block);
                }
             }
             else
             {
                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
             }
         }
     }))
{
}

Baseband_Stream::~Baseband_Stream()
{
    m_exit_signal.set_value();
    process_thread.join();
    m_next_transform.clear();
}

void rotate_90(unsigned char *buf, uint32_t len)
/* 90 rotation is 1+0j, 0+1j, -1+0j, 0-1j
   or [0, 1, -3, 2, -4, -5, 7, -6] */
{
    uint32_t i = 0;
    unsigned char tmp = 0;
    for (i=0; i<len; i+=8) {
        /* uint8_t negation = 255 - x */
        tmp = 255 - buf[i+3];
        buf[i+3] = buf[i+2];
        buf[i+2] = tmp;

        buf[i+4] = 255 - buf[i+4];
        buf[i+5] = 255 - buf[i+5];

        tmp = 255 - buf[i+6];
        buf[i+6] = buf[i+7];
        buf[i+7] = tmp;
    }
}

void Baseband_Stream::rotate_90(unsigned char *buf, uint32_t len)
/* 90 rotation is 1+0j, 0+1j, -1+0j, 0-1j
   or [0, 1, -3, 2, -4, -5, 7, -6] */
{
    uint32_t i = 0;
    unsigned char tmp = 0;
    for (i=0; i<len; i+=8) {
        /* uint8_t negation = 255 - x */
        tmp = 255 - buf[i+3];
        buf[i+3] = buf[i+2];
        buf[i+2] = tmp;

        buf[i+4] = 255 - buf[i+4];
        buf[i+5] = 255 - buf[i+5];

        tmp = 255 - buf[i+6];
        buf[i+6] = buf[i+7];
        buf[i+7] = tmp;
    }
}

void Baseband_Stream::load_buffer_and_rotate90(char* iq, size_t n) // must be a multiple of 2 [1 real, 2 imag]
{
    BBP_Block* block = m_BBP_Buffer_Pool->next(); // thread safe

    if(block != nullptr)
    {
        rotate_90((unsigned char*)iq, n);

        for (size_t i = 0, p = 0; i < n; i += 2, p++)
        {
            block->points[p].real((RADIO_DATA_TYPE)iq[i] - 127.4);
            block->points[p].imag((RADIO_DATA_TYPE)iq[i + 1] - 127.4);
            // std::cerr << block->points[p].real() << "  " << block->points[p].imag() << std::endl;
        }

        block->n_points = n / 2; // set the number of points stored in the buffer block

        m_buffer.push(block);
    }
    else
    {
        std::cout << "Dude not loading that buffer ... ran out of frames..." << std::endl;
    }
}

void Baseband_Stream::load_buffer(char* iq, size_t n) // must be a multiple of 2 [1 real, 2 imag]
{
    BBP_Block* block = m_BBP_Buffer_Pool->next(); // thread safe

    if(block != nullptr)
    {
        for (size_t i = 0, p = 0; i < n; i += 2, p++)
        {
            block->points[p].real(((RADIO_DATA_TYPE)iq[i]     - 127.4));//*1000000000.0);
            block->points[p].imag(((RADIO_DATA_TYPE)iq[i + 1] - 127.4));//*1000000000.0);
        }

        block->n_points = n / 2; // set the number of points stored in the buffer block

        m_buffer.push(block);
    }
    else
    {
        std::cout << "Dude not loading that buffer ... ran out of frames..." << std::endl;
    }
}


void Baseband_Stream::load_buffer(BBP_Block* block)
{
    m_buffer.push(block);
}

void Baseband_Stream::add_next_buffer(Baseband_Stream* stream)
{
    m_next_transform.push_back(stream);
}


