/// @file SDR/RadioNode/src/libraries/common/Baseband_Stream.h


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

/// \brief Baseband Stream for passing complex data, and processing.
/// \details Allows the user to branch the data across multiple streams for different processing.
class Baseband_Stream
{
    public:
        /// \brief Constructor
        /// \param post_cb Callback that is triggered after data is processed.
        /// \param BBP_Buffer_Pool Buffer pool to return buffers to after processing.
        Baseband_Stream(std::function<void(BBP_Block*)> post_cb,
                        SafeBufferPoolQueue* BBP_Buffer_Pool );

        /// \brief Destructor
        ~Baseband_Stream();

        /// \brief Load a buffer from raw sample stream, and rotate it 90 degrees.
        /// \param iq complex byte stream alternating In-Phase and Quad data
        /// \param n total number of samples in array
        void load_buffer_and_rotate90(char* iq, size_t n); // must be a multiple of 2 [1 real, 2 imag]

        /// \brief Load a buffer from raw sample stream
        /// \param iq complex byte stream alternating In-Phase and Quad data
        /// \param n total number of samples in array
        void load_buffer(char* iq, size_t n); // must be a multiple of 2 [1 real, 2 imag]

        /// \brief Load a buffer from an existing Block
        /// \param block Baseband Block
        void load_buffer(BBP_Block* block);

        /// \brief Push another downstream location that data will go when post processed.
        /// \param stream Next downstream Baseband_Stream
        void add_next_buffer(Baseband_Stream* stream);

        /// \brief Get the number of buffers waiting
        /// \return number of blocks in waiting
        size_t get_blocks_waiting() const
        {
            return m_buffer.get_available_blocks();
        }

    private:
        /// \brief Rotate the stream 90 degrees
        /// \param buf IQ buffer stream.
        /// \param len Length of samples
        void rotate_90(unsigned char *buf, uint32_t len);

        SafeBufferPoolQueue m_buffer;   ///< Internal buffer queue
        std::function<void(BBP_Block*)> m_post_cb;  ///< Post callback funtion
        std::vector<Baseband_Stream*> m_next_transform; ///< Vector of downstream Streams
        SafeBufferPoolQueue* m_BBP_Buffer_Pool; ///< Return Buffer queue

        // Process Thread
        std::promise<void> m_exit_signal;   ///< Exit Signal for threads
        std::future<void> m_future_obj;     ///< Future Object used for terminating Threads
        std::thread process_thread;         ///< Baseband Processing thread
};

#endif
