/// @file SDR/RadioNode/src/libraries/crc/crc.h


#ifndef RADIONODE_CRC8_H
#define RADIONODE_CRC8_H

#include <stdlib.h>
#include <stdint.h>

/// Basic CRC interface to generate 8 bit CRC calculations
class crc8
{
    public:
        /// constructor
        crc8();

        /// destructor
        ~crc8();

        /// Calculate a CRC from a byte array
        ///
        /// \param[in] message  Byte array
        /// \param[in] length   Length of the byte array
        /// \return CRC for the given array
        uint8_t getCRC(uint8_t message[], size_t length);

    private:
        /// Get the individual CRC for each byte
        ///
        /// \param[in] val  input value
        /// \return Returns the crc for the given byte.
        uint8_t getCRCForByte(uint8_t val);

    private:
        const uint8_t CRC7_POLY = 0x91;     ///< CRC7 Poly
        uint8_t CRCTable[256];              ///< Table of crc bytes for given crc
};


#endif //RADIONODE_CRC8_H
