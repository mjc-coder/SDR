//
// Created by Micheal Cowan on 9/14/19.
//

#ifndef RADIONODE_CRC8_H
#define RADIONODE_CRC8_H

#include <stdlib.h>
#include <stdint.h>

class crc8
{
    public:
        crc8();

        ~crc8();

        uint8_t getCRC(uint8_t message[], size_t length);

    private:
        uint8_t getCRCForByte(uint8_t val);

    private:
        const uint8_t CRC7_POLY = 0x91;
        uint8_t CRCTable[256];
};


#endif //RADIONODE_CRC8_H
