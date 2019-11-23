/// @file SDR/RadioNode/src/libraries/crc/crc.cpp


#include <crc/crc8.h>
#include <iostream>

crc8::crc8()
{
    // fill an array with CRC values of all 256 possible bytes
    for (size_t i = 0; i < 256; i++)
    {
        CRCTable[i] = getCRCForByte(i);
    }
}

crc8::~crc8()
{

}

uint8_t crc8::getCRCForByte(uint8_t val)
{
    for (size_t j = 0; j < 8; j++)
    {
        if (val & 1)
            val ^= CRC7_POLY;
        val >>= 1;
    }

    return val;
}



uint8_t crc8::getCRC(uint8_t message[], size_t length)
{
    uint8_t crc = 0;

    for (size_t i = 0; i < length; i++)
    {
        crc = CRCTable[crc ^ message[i]];
    }
    return crc;
}

