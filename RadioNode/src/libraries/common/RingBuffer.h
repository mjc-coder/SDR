/// @file SDR/RadioNode/src/libraries/common/RingBuffer.h


#ifndef RADIONODE_RINGBUFFER_H
#define RADIONODE_RINGBUFFER_H

#include <string>
#include <stdexcept>
#include <mutex>
#include <iostream>

/// \brief Templated Ring Buffer class
/// \tparam T Buffer Type must implement copy constructor.
template<typename T>
class RingBuffer
{
public:
    /// \brief Constructor
    /// \param size Number of elements to create ring buffer with.
    RingBuffer(int size);

    /// \brief Destructor
    ~RingBuffer();

    /// \brief Check if empty
    /// \returns true if empty, false otherwise.
    bool empty() const
    {
        return m_count == 0;
    }

    /// \brief Check if full
    /// \returns true if full, false otherwise.
    bool full() const
    {
        return m_count == m_size;
    }

    /// \brief Append an element
    /// \returns true if successful, false otherwise
    bool append(const T&);

    /// \brief append a Group of elements
    /// @param[in] elem  list of elements
    /// @param[in] len   number of elements to append
    /// \returns number of elements added
    size_t append(const T* elem, size_t len);

    /// \brief pops first element
    /// \returns if element was removed
    bool remove();

    /// \brief removes N elements from the front
    /// \param number of elements to remove
    /// \returns true if successful, false otherwise
    bool remove(size_t len);

    /// \brief Get the number of elements in the buffer
    /// \return zero based number of elements.
    size_t count() const
    {
        return m_count;
    }

    /// \brief Get the size of the ring buffer
    /// \return Max number of elements in the ring buffer
    size_t max() const
    {
        return m_size;
    }

    /// \brief Get the number of elements in the right buffer
    /// \return Number of elements in the buffer.
    size_t num_elements() const
    {
        return m_count+1;
    }

    /// \brief Get the value at the given position
    /// \param pos The given index in the array. Must be less than the max, and is 0 based.
    /// \return Returns the value.
    T value(int pos) const
    {
        return m_data[(m_front + pos)%m_size];
    }

    /// \brief Reset the buffer and clear the values.
    void reset()
    {
        m_lock.lock();
        m_count = 0;
        m_front = 0;
        memset(m_data, 0, m_size*sizeof(T));
        m_lock.unlock();
    }

private:
    RingBuffer() = delete;  ///< disallow the basic constructor
    const int m_size;   ///< The max size of the Ring Buffer.
    int m_count;    ///< Current count of items in buffer.
    int m_front;    ///< Front index of the buffer.
    std::mutex m_lock;  ///< Buffer Lock for thread safety
    T* m_data;  ///< Buffer Data array.
};


/// \brief Constructor
/// \tparam T Buffer Type
/// \param sz Size of the ring buffer
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

/// \brief Destructor
/// \tparam T Buffer Type
template<typename T>
RingBuffer<T>::~RingBuffer()
{
    delete[] m_data;
}

/// \brief Append an element to the buffer.
/// \tparam T Buffer Type
/// \param t value to append.
/// \return true if successful, false otherwise.
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

/// \brief Append multiple elements from an array
/// \tparam T   Buffer Type
/// \param elem Element array of type Buffer Type
/// \param len  Number of elements ot append
/// \return Number of elements actually appended to the buffer.
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

/// \brief Remove first element in the Buffer
/// \tparam T Buffer Type
/// \return True if successful, false otherwise.
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

/// \brief Remove one or more elements from the buffer.
/// \tparam T Buffer Type
/// \param len Number of elements to remove
/// \return true if all elements were removed, falst if it failed.
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
