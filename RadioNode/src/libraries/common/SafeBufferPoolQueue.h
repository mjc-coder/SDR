/// @file SDR/RadioNode/src/libraries/common/SafeBufferPoolQueue.h


#ifndef RADIONODE_SAFEBUFFERBOOLQUEUE_H
#define RADIONODE_SAFEBUFFERBOOLQUEUE_H

#include <common/Common_Deffinitions.h>
#include <common/BBP_Block.h>
#include <queue>
#include <mutex>

struct BBP_Block; ///< Predeclaration of the BBP Block

/// \brief A thread safe BBP Block Buffer Queue
class SafeBufferPoolQueue
{
private:
    std::queue<BBP_Block*> q;   ///< Internal Queue of BBP Blocks
    std::mutex m; ///< Mutex for thread safety

public:

    /// \brief Constructor
    /// \param size Number of buffers to initialize the Queue with.
    SafeBufferPoolQueue(size_t size = 0);

    /// \brief Destructor
    ~SafeBufferPoolQueue();

    /// \brief Push an element onto the back of the Queue
    /// \param elem BBP Block to append the the queue.
    void push(BBP_Block* elem);

    /// \brief Pop the next available element from the Queue.
    /// \return Pointer to the queue element.  Element is removed from Queue.
    BBP_Block* next();

    /// \brief Get total number of available elements on the queue.
    /// \return Number of elements.
    size_t get_available_blocks() const;
};


#endif //RADIONODE_SAFEBUFFERBOOLQUEUE_H
