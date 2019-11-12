//
// Created by Micheal Cowan on 9/13/19.
//

#include <common/confidence_counter.h>
#include <iostream>

confidence_counter::confidence_counter(int32_t max, int32_t low_threshold, int32_t high_threshold)
: m_counter(max/2) // start low confidence side
, m_low_threshold(low_threshold)
, m_high_threshold(high_threshold)
, m_max(max)
{
}

confidence_counter::~confidence_counter()
{
}

int32_t confidence_counter::value() const
{
    return m_counter;
}

bool confidence_counter::high_confidence() const
{
    if(m_counter >= m_high_threshold)
    {
        return true;
    }
    return false;
}

bool confidence_counter::low_confidence() const
{
    if(m_counter < m_low_threshold)
    {
        return true;
    }
    return false;
}

void confidence_counter::operator ++()
{
    m_lock.lock();
    m_counter++;
    if(m_counter >= m_max)
    {
        m_counter = m_max;
    }
    m_lock.unlock();
}

void confidence_counter::operator--()
{
    m_lock.lock();
    m_counter--;
    if(m_counter <= 0)
    {
        m_counter = 0;
    }
    m_lock.unlock();
}
void confidence_counter::operator++(int)
{
    m_lock.lock();
    m_counter++;
    if(m_counter >= m_max)
    {
        m_counter = m_max;
    }
    m_lock.unlock();
}

void confidence_counter::operator--(int)
{
    m_lock.lock();
    m_counter--;
    if(m_counter <= 0)
    {
        m_counter = 0;
    }
    m_lock.unlock();
}
