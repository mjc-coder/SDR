//
// Created by Micheal Cowan on 2019-07-5.
//

#ifndef BASEBAND_STREAM_H   
#define BASEBAND_STREAM_H

#include <common/BBP_Block.h>
#include <common/SafeBufferPoolQueue.h>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>
#include <future>
#include <queue>
#include <chrono>
#include <string.h>

struct BBP_Block;
class SafeBufferPoolQueue;

class Baseband_Stream
{
    public:
        Baseband_Stream(std::function<void(BBP_Block*)> post_cb,
                        SafeBufferPoolQueue* BBP_Buffer_Pool );

        ~Baseband_Stream();

        void load_buffer_and_rotate90(char* iq, size_t n); // must be a multiple of 2 [1 real, 2 imag]

        void load_buffer(char* iq, size_t n); // must be a multiple of 2 [1 real, 2 imag]

        void load_buffer(BBP_Block* block);

        void add_next_buffer(Baseband_Stream* stream);

        size_t get_blocks_waiting() const
        {
            return m_buffer.get_available_blocks();
        }

    private:
        void rotate_90(unsigned char *buf, uint32_t len);

        SafeBufferPoolQueue m_buffer;
        std::function<void(BBP_Block*)> m_post_cb;
        std::vector<Baseband_Stream*> m_next_transform;
        SafeBufferPoolQueue* m_BBP_Buffer_Pool;

        // Process Thread
        std::promise<void> m_exit_signal;
        std::future<void> m_future_obj;
        std::thread process_thread;
};

#endif
