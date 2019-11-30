/// @file SDR/RadioNode/src/libraries/common/confidence_counter.h


#ifndef RADIONODE_CONFIDENCE_COUNTER_H
#define RADIONODE_CONFIDENCE_COUNTER_H

#include <stdlib.h>
#include <mutex>

/// \brief A Confidence counter is a binary fuzzy logic switch.  Allows for variable threshold values
/// to switch from high to low confidence, and a variable size of the counter.
class confidence_counter
{
public:
    /// \brief Constructor
    /// \param max Confidence counter size
    /// \param low_threshold lower end of the threshold
    /// \param high_threshold higher end of the threshold
    confidence_counter(int32_t max, int32_t low_threshold, int32_t high_threshold);

    /// \brief Destructor
    ~confidence_counter();

    /// \brief Get the current value of the counter.
    /// \return The current count.
    int32_t value() const;

    /// \brief Checks if the confidence counter is in the high threshold range.
    /// \return True if high confidence, false if otherwise.
    bool high_confidence() const;

    /// \brief Checks if the confidence counter is in the low threshold range.
    /// \return True if low confidence, false if otherwise.
    bool low_confidence() const;

    /// \brief Increment counter.
    void operator++();

    /// \brief Decrement counter.
    void operator--();

    /// \brief Increment the counter by one.
    void operator++(int32_t);

    /// \brief Decrement the counter by one.
    void operator--(int32_t);

    /// \brief Decrement the counter by a value
    /// \param dec  Decrement by this value
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

    /// \brief Increment the counter by a value
    /// \param inc Increment by this value
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

    /// \brief Reset all the counters to the middle value.
    void reset()
    {
        m_lock.lock();
        m_counter = m_max/2;
        m_lock.unlock();
    }

private:
    int32_t m_counter; ///< Counter value
    int32_t m_low_threshold;    ///< low threshold value
    int32_t m_high_threshold;   ///< high threshold value
    int32_t m_max;  /// Max confidence counter value
    std::mutex m_lock;  ///< counter lock for thread safety
};


#endif //RADIONODE_CONFIDENCE_COUNTER_H
