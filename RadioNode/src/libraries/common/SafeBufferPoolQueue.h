//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_SAFEBUFFERBOOLQUEUE_H
#define RADIONODE_SAFEBUFFERBOOLQUEUE_H

#include <common/Common_Deffinitions.h>
#include <common/BBP_Block.h>
#include <queue>
#include <mutex>

struct BBP_Block;
class SafeBufferPoolQueue
{
private:
    std::queue<BBP_Block*> q;
    std::mutex m;

public:

    SafeBufferPoolQueue(size_t size = 0);

    ~SafeBufferPoolQueue();

    void push(BBP_Block* elem);

    BBP_Block* next();

    size_t get_available_blocks() const;
};


#endif //RADIONODE_SAFEBUFFERBOOLQUEUE_H
