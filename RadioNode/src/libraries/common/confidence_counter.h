//
// Created by Micheal Cowan on 9/13/19.
//

#ifndef RADIONODE_CONFIDENCE_COUNTER_H
#define RADIONODE_CONFIDENCE_COUNTER_H

#include <stdlib.h>
#include <mutex>

class confidence_counter
{
public:
    confidence_counter(int32_t max, int32_t low_threshold, int32_t high_threshold);

    ~confidence_counter();

    int32_t value() const;

    bool high_confidence() const;

    bool low_confidence() const;

    void operator++();

    void operator--();
    void operator++(int32_t);

    void operator--(int32_t);

    void decrement(int32_t dec)
    {
        m_lock.lock();
        m_counter-=dec;
        if(m_counter < 0)
        {
            m_counter = 0;
        }
        m_lock.unlock();
    }

    void increment(int32_t inc)
    {
        m_lock.lock();
        m_counter+=inc;
        if(m_counter > m_max)
        {
            m_counter = m_max;
        }
        m_lock.unlock();
    }

    void reset()
    {
        m_lock.lock();
        m_counter = m_max/2;
        m_lock.unlock();
    }

private:
    int32_t m_counter;
    int32_t m_low_threshold;
    int32_t m_high_threshold;
    int32_t m_max;
    std::mutex m_lock;
};


#endif //RADIONODE_CONFIDENCE_COUNTER_H
