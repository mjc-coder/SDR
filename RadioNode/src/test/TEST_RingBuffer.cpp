//
// Created by Micheal Cowan on 9/14/19.
//

#include <common/RingBuffer.h>
#include <gtest/gtest.h> // googletest header file

TEST(RingBuffer, Constructor)
{
    RingBuffer<uint8_t> buf(10);

    ASSERT_TRUE(buf.empty());
    ASSERT_TRUE(!buf.full());
}


TEST(RingBuffer, append_remove)
{
    RingBuffer<uint8_t> buf(10);

    for(uint8_t i = 0; i < 100; i++)
    {
        ASSERT_TRUE(buf.append(i));
        ASSERT_TRUE(buf.remove());
    }
}

TEST(RingBuffer, append_remove_errors)
{
    RingBuffer<uint8_t> buf(10);

    ASSERT_FALSE(buf.remove());

    for(uint8_t i = 0; i < 10; i++)
    {
        ASSERT_TRUE(buf.append(i));
    }

    ASSERT_FALSE(buf.append(0));
}

TEST(RingBuffer, append_remove_list)
{
    RingBuffer<uint8_t> buf(10);
    uint8_t buffer[10] = {0};

    ASSERT_FALSE(buf.remove());

    ASSERT_TRUE(buf.append(buffer, 10));
    ASSERT_TRUE(buf.full());

    ASSERT_FALSE(buf.append(0));
    ASSERT_TRUE(buf.remove(10));
    ASSERT_TRUE(buf.empty());
}
