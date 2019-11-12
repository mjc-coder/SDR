//
// Created by Micheal Cowan on 9/7/19.
//

#include <common/SafeBufferPoolQueue.h>


SafeBufferPoolQueue::SafeBufferPoolQueue(size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        BBP_Block* b = new BBP_Block();
        b->reset();
        q.push(b);
    }
}

SafeBufferPoolQueue::~SafeBufferPoolQueue()
{
    m.lock();
    while(q.size() > 0)
    {
        delete q.front();
        q.pop();
    }
    m.unlock();
}

void SafeBufferPoolQueue::push(BBP_Block* elem)
{
    m.lock();
    if(elem != nullptr)
    {
        q.push(elem);
    }
    m.unlock();
}

BBP_Block* SafeBufferPoolQueue::next() {

    BBP_Block* elem = nullptr;

    m.lock();
    if(!q.empty())
    {
        elem = q.front();
        q.pop();
    }
    m.unlock();

    return elem;
}

size_t SafeBufferPoolQueue::get_available_blocks() const
{
    return q.size();
}



