//
// Created by Micheal Cowan on 10/10/19.
//

#ifndef RADIONODE_RINGBUFFER_H
#define RADIONODE_RINGBUFFER_H


#include <string>
#include <stdexcept>
#include <mutex>
#include <iostream>
#include <string.h>
/*
   T must implement operator=, copy ctor
*/
template<typename T>
class RingBuffer
{
public:
    /// constructor
    RingBuffer(int size);

    /// destructor
    ~RingBuffer();

    /// Check if empty
    bool empty() const
    {
        return m_count == 0;
    }

    /// check if full
    bool full() const
    {
        return m_count == m_size;
    }

    /// Append an element
    bool append(const T&);

    /// append a Group of elements
    /// @param[in] elem  list of elements
    /// @param[in] len   number of elements to add
    ///
    /// \return number of elements added
    size_t append(const T* elem, size_t len);

    /// pops first element
    bool remove();

    /// removes N elements from the front
    bool remove(size_t len);

    size_t count() const
    {
        return m_count;
    }

    size_t max() const
    {
        return m_size;
    }

    size_t num_elements() const
    {
        return m_count+1;
    }


    T value(int pos) const
    {
        return m_data[(m_front + pos)%m_size];
    }

    void reset()
    {
        m_lock.lock();
        m_count = 0;
        m_front = 0;
        memset(m_data, 0, m_size*sizeof(T));
        m_lock.unlock();
    }

private:
    RingBuffer() = delete;
    const int m_size;
    int m_count;
    int m_front;
    std::mutex m_lock;
    T* m_data;
};

template<typename T>
RingBuffer<T>::RingBuffer(int sz): m_size(sz)
{
    if (sz==0) {
        throw std::invalid_argument("size cannot be zero");
    }
    m_data = new T[sz];
    memset((void*)m_data, 0 , sz*sizeof(T));
    m_front = 0;
    m_count = 0;

}


template<typename T>
RingBuffer<T>::~RingBuffer()
{
    delete[] m_data;
}

// returns true if add was successful, false if the buffer is already full
template<typename T>
bool RingBuffer<T>::append(const T &t)
{
    bool returnVal = false;

    if ( !full() )
    {
        // find index where insert will occur
        m_lock.lock();
        m_data[(m_front + m_count) % m_size] = t;
        m_count++;
        m_lock.unlock();
        returnVal = true;
    }
    return returnVal;
}

template<typename T>
size_t RingBuffer<T>::append(const T* elem, size_t len)
{
    if(!full())
    {
        // find index where insert will occur
        for(size_t i = 0; i < len; i++)
        {
            if( !append(elem[i]) )
            {
                return i;
            }
        }
    }
    return len;
}

// returns true if there is something to remove, false otherwise
template<typename T>
bool RingBuffer<T>::remove()
{
    bool returnVal = false;

    if ( !empty() )
    {
        m_lock.lock();
        if(m_front == m_size)
        {
            m_front = 0;
        }
        m_data[m_front] = 0;
        m_front = (m_front == m_size) ? 0 : m_front + 1; // reset front
        m_count--;
        m_lock.unlock();
        returnVal = true;
    }

    return returnVal;
}

template<typename T>
bool RingBuffer<T>::remove(size_t len)
{
    if ( !empty() )
    {
        for(size_t i = 0; i < len; i++)
        {
            if(!remove())
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

#endif //RADIONODE_RINGBUFFER_H
