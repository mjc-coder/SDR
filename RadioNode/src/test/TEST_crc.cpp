//
// Created by Micheal Cowan on 9/14/19.
//

#include <crc/crc8.h>
#include <gtest/gtest.h> // googletest header file

TEST(CRC, crc_check)
{
    uint8_t message[3] = {0x83, 0x01, 0x00};

    crc8 crc_calculator;
    uint8_t calc = crc_calculator.getCRC(message, 2);

    ASSERT_EQ(0x17, calc);
}

