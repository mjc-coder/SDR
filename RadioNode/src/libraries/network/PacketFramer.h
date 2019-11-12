//
// Created by Micheal Cowan on 9/13/19.
//

#ifndef RADIONODE_PACKETFRAMER_H
#define RADIONODE_PACKETFRAMER_H

#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <crc/crc8.h>
#include <iostream>
#include <common/RingBuffer.h>
#include <string.h>

// Packet Description
// <unique word 16 bit> |  <length 8 bit> | <crc 8 bit>
// < data > [no more than 250 bytes to keep it simple]
//
// Converted to serial bits
// 32 + 250*8 = 2032 bytes of single bits 0 and 1


const uint8_t PACKET_FRAMER_UNIQUE_WORD[16] =
        {1,1,1,1,1,0,1,0,
         1,1,0,0,1,1,1,0};

template <class data_type>
class PacketFramer
{
private:
    bool m_crc_enable;
    data_type* m_pkt;
    crc8 crcCalculator;

public:
    static constexpr size_t MAX_BYTES_DATA = 250;
    static constexpr size_t UNIQUE_WORD_SIZE_BIT = 16;
    static constexpr size_t UNIQUE_WORD_SIZE_BYTE = 2;
    static constexpr size_t UNIQUE_WORD_POS = 0;
    static constexpr size_t LENGTH_SIZE_BIT = 8;
    static constexpr size_t LENGTH_SIZE_BYTE = 1;
    static constexpr size_t LENGTH_POS = UNIQUE_WORD_POS + UNIQUE_WORD_SIZE_BIT;
    static constexpr size_t CRC_SIZE_BIT = 8;
    static constexpr size_t CRC_SIZE_BYTE = 1;
    static constexpr size_t CRC_POS = LENGTH_POS + LENGTH_SIZE_BIT;
    static constexpr size_t DATA_SIZE_BIT = MAX_BYTES_DATA*8;
    static constexpr size_t DATA_SIZE_BYTE = MAX_BYTES_DATA;
    static constexpr size_t DATA_POS = CRC_POS + CRC_SIZE_BIT;
    static constexpr size_t MAX_PKT_SIZE_BITS = DATA_POS + DATA_SIZE_BIT;
    static constexpr size_t MAX_PKT_SIZE_BYTE = MAX_PKT_SIZE_BITS/8;

public:
    const data_type binary_array[256][8]  =
            {
                    {0,0,0,0,0,0,0,0},
                    {0,0,0,0,0,0,0,1},
                    {0,0,0,0,0,0,1,0},
                    {0,0,0,0,0,0,1,1},
                    {0,0,0,0,0,1,0,0},
                    {0,0,0,0,0,1,0,1},
                    {0,0,0,0,0,1,1,0},
                    {0,0,0,0,0,1,1,1},
                    {0,0,0,0,1,0,0,0},
                    {0,0,0,0,1,0,0,1},
                    {0,0,0,0,1,0,1,0},
                    {0,0,0,0,1,0,1,1},
                    {0,0,0,0,1,1,0,0},
                    {0,0,0,0,1,1,0,1},
                    {0,0,0,0,1,1,1,0},
                    {0,0,0,0,1,1,1,1},
                    {0,0,0,1,0,0,0,0},
                    {0,0,0,1,0,0,0,1},
                    {0,0,0,1,0,0,1,0},
                    {0,0,0,1,0,0,1,1},
                    {0,0,0,1,0,1,0,0},
                    {0,0,0,1,0,1,0,1},
                    {0,0,0,1,0,1,1,0},
                    {0,0,0,1,0,1,1,1},
                    {0,0,0,1,1,0,0,0},
                    {0,0,0,1,1,0,0,1},
                    {0,0,0,1,1,0,1,0},
                    {0,0,0,1,1,0,1,1},
                    {0,0,0,1,1,1,0,0},
                    {0,0,0,1,1,1,0,1},
                    {0,0,0,1,1,1,1,0},
                    {0,0,0,1,1,1,1,1},
                    {0,0,1,0,0,0,0,0},
                    {0,0,1,0,0,0,0,1},
                    {0,0,1,0,0,0,1,0},
                    {0,0,1,0,0,0,1,1},
                    {0,0,1,0,0,1,0,0},
                    {0,0,1,0,0,1,0,1},
                    {0,0,1,0,0,1,1,0},
                    {0,0,1,0,0,1,1,1},
                    {0,0,1,0,1,0,0,0},
                    {0,0,1,0,1,0,0,1},
                    {0,0,1,0,1,0,1,0},
                    {0,0,1,0,1,0,1,1},
                    {0,0,1,0,1,1,0,0},
                    {0,0,1,0,1,1,0,1},
                    {0,0,1,0,1,1,1,0},
                    {0,0,1,0,1,1,1,1},
                    {0,0,1,1,0,0,0,0},
                    {0,0,1,1,0,0,0,1},
                    {0,0,1,1,0,0,1,0},
                    {0,0,1,1,0,0,1,1},
                    {0,0,1,1,0,1,0,0},
                    {0,0,1,1,0,1,0,1},
                    {0,0,1,1,0,1,1,0},
                    {0,0,1,1,0,1,1,1},
                    {0,0,1,1,1,0,0,0},
                    {0,0,1,1,1,0,0,1},
                    {0,0,1,1,1,0,1,0},
                    {0,0,1,1,1,0,1,1},
                    {0,0,1,1,1,1,0,0},
                    {0,0,1,1,1,1,0,1},
                    {0,0,1,1,1,1,1,0},
                    {0,0,1,1,1,1,1,1},
                    {0,1,0,0,0,0,0,0},
                    {0,1,0,0,0,0,0,1},
                    {0,1,0,0,0,0,1,0},
                    {0,1,0,0,0,0,1,1},
                    {0,1,0,0,0,1,0,0},
                    {0,1,0,0,0,1,0,1},
                    {0,1,0,0,0,1,1,0},
                    {0,1,0,0,0,1,1,1},
                    {0,1,0,0,1,0,0,0},
                    {0,1,0,0,1,0,0,1},
                    {0,1,0,0,1,0,1,0},
                    {0,1,0,0,1,0,1,1},
                    {0,1,0,0,1,1,0,0},
                    {0,1,0,0,1,1,0,1},
                    {0,1,0,0,1,1,1,0},
                    {0,1,0,0,1,1,1,1},
                    {0,1,0,1,0,0,0,0},
                    {0,1,0,1,0,0,0,1},
                    {0,1,0,1,0,0,1,0},
                    {0,1,0,1,0,0,1,1},
                    {0,1,0,1,0,1,0,0},
                    {0,1,0,1,0,1,0,1},
                    {0,1,0,1,0,1,1,0},
                    {0,1,0,1,0,1,1,1},
                    {0,1,0,1,1,0,0,0},
                    {0,1,0,1,1,0,0,1},
                    {0,1,0,1,1,0,1,0},
                    {0,1,0,1,1,0,1,1},
                    {0,1,0,1,1,1,0,0},
                    {0,1,0,1,1,1,0,1},
                    {0,1,0,1,1,1,1,0},
                    {0,1,0,1,1,1,1,1},
                    {0,1,1,0,0,0,0,0},
                    {0,1,1,0,0,0,0,1},
                    {0,1,1,0,0,0,1,0},
                    {0,1,1,0,0,0,1,1},
                    {0,1,1,0,0,1,0,0},
                    {0,1,1,0,0,1,0,1},
                    {0,1,1,0,0,1,1,0},
                    {0,1,1,0,0,1,1,1},
                    {0,1,1,0,1,0,0,0},
                    {0,1,1,0,1,0,0,1},
                    {0,1,1,0,1,0,1,0},
                    {0,1,1,0,1,0,1,1},
                    {0,1,1,0,1,1,0,0},
                    {0,1,1,0,1,1,0,1},
                    {0,1,1,0,1,1,1,0},
                    {0,1,1,0,1,1,1,1},
                    {0,1,1,1,0,0,0,0},
                    {0,1,1,1,0,0,0,1},
                    {0,1,1,1,0,0,1,0},
                    {0,1,1,1,0,0,1,1},
                    {0,1,1,1,0,1,0,0},
                    {0,1,1,1,0,1,0,1},
                    {0,1,1,1,0,1,1,0},
                    {0,1,1,1,0,1,1,1},
                    {0,1,1,1,1,0,0,0},
                    {0,1,1,1,1,0,0,1},
                    {0,1,1,1,1,0,1,0},
                    {0,1,1,1,1,0,1,1},
                    {0,1,1,1,1,1,0,0},
                    {0,1,1,1,1,1,0,1},
                    {0,1,1,1,1,1,1,0},
                    {0,1,1,1,1,1,1,1},
                    {1,0,0,0,0,0,0,0},
                    {1,0,0,0,0,0,0,1},
                    {1,0,0,0,0,0,1,0},
                    {1,0,0,0,0,0,1,1},
                    {1,0,0,0,0,1,0,0},
                    {1,0,0,0,0,1,0,1},
                    {1,0,0,0,0,1,1,0},
                    {1,0,0,0,0,1,1,1},
                    {1,0,0,0,1,0,0,0},
                    {1,0,0,0,1,0,0,1},
                    {1,0,0,0,1,0,1,0},
                    {1,0,0,0,1,0,1,1},
                    {1,0,0,0,1,1,0,0},
                    {1,0,0,0,1,1,0,1},
                    {1,0,0,0,1,1,1,0},
                    {1,0,0,0,1,1,1,1},
                    {1,0,0,1,0,0,0,0},
                    {1,0,0,1,0,0,0,1},
                    {1,0,0,1,0,0,1,0},
                    {1,0,0,1,0,0,1,1},
                    {1,0,0,1,0,1,0,0},
                    {1,0,0,1,0,1,0,1},
                    {1,0,0,1,0,1,1,0},
                    {1,0,0,1,0,1,1,1},
                    {1,0,0,1,1,0,0,0},
                    {1,0,0,1,1,0,0,1},
                    {1,0,0,1,1,0,1,0},
                    {1,0,0,1,1,0,1,1},
                    {1,0,0,1,1,1,0,0},
                    {1,0,0,1,1,1,0,1},
                    {1,0,0,1,1,1,1,0},
                    {1,0,0,1,1,1,1,1},
                    {1,0,1,0,0,0,0,0},
                    {1,0,1,0,0,0,0,1},
                    {1,0,1,0,0,0,1,0},
                    {1,0,1,0,0,0,1,1},
                    {1,0,1,0,0,1,0,0},
                    {1,0,1,0,0,1,0,1},
                    {1,0,1,0,0,1,1,0},
                    {1,0,1,0,0,1,1,1},
                    {1,0,1,0,1,0,0,0},
                    {1,0,1,0,1,0,0,1},
                    {1,0,1,0,1,0,1,0},
                    {1,0,1,0,1,0,1,1},
                    {1,0,1,0,1,1,0,0},
                    {1,0,1,0,1,1,0,1},
                    {1,0,1,0,1,1,1,0},
                    {1,0,1,0,1,1,1,1},
                    {1,0,1,1,0,0,0,0},
                    {1,0,1,1,0,0,0,1},
                    {1,0,1,1,0,0,1,0},
                    {1,0,1,1,0,0,1,1},
                    {1,0,1,1,0,1,0,0},
                    {1,0,1,1,0,1,0,1},
                    {1,0,1,1,0,1,1,0},
                    {1,0,1,1,0,1,1,1},
                    {1,0,1,1,1,0,0,0},
                    {1,0,1,1,1,0,0,1},
                    {1,0,1,1,1,0,1,0},
                    {1,0,1,1,1,0,1,1},
                    {1,0,1,1,1,1,0,0},
                    {1,0,1,1,1,1,0,1},
                    {1,0,1,1,1,1,1,0},
                    {1,0,1,1,1,1,1,1},
                    {1,1,0,0,0,0,0,0},
                    {1,1,0,0,0,0,0,1},
                    {1,1,0,0,0,0,1,0},
                    {1,1,0,0,0,0,1,1},
                    {1,1,0,0,0,1,0,0},
                    {1,1,0,0,0,1,0,1},
                    {1,1,0,0,0,1,1,0},
                    {1,1,0,0,0,1,1,1},
                    {1,1,0,0,1,0,0,0},
                    {1,1,0,0,1,0,0,1},
                    {1,1,0,0,1,0,1,0},
                    {1,1,0,0,1,0,1,1},
                    {1,1,0,0,1,1,0,0},
                    {1,1,0,0,1,1,0,1},
                    {1,1,0,0,1,1,1,0},
                    {1,1,0,0,1,1,1,1},
                    {1,1,0,1,0,0,0,0},
                    {1,1,0,1,0,0,0,1},
                    {1,1,0,1,0,0,1,0},
                    {1,1,0,1,0,0,1,1},
                    {1,1,0,1,0,1,0,0},
                    {1,1,0,1,0,1,0,1},
                    {1,1,0,1,0,1,1,0},
                    {1,1,0,1,0,1,1,1},
                    {1,1,0,1,1,0,0,0},
                    {1,1,0,1,1,0,0,1},
                    {1,1,0,1,1,0,1,0},
                    {1,1,0,1,1,0,1,1},
                    {1,1,0,1,1,1,0,0},
                    {1,1,0,1,1,1,0,1},
                    {1,1,0,1,1,1,1,0},
                    {1,1,0,1,1,1,1,1},
                    {1,1,1,0,0,0,0,0},
                    {1,1,1,0,0,0,0,1},
                    {1,1,1,0,0,0,1,0},
                    {1,1,1,0,0,0,1,1},
                    {1,1,1,0,0,1,0,0},
                    {1,1,1,0,0,1,0,1},
                    {1,1,1,0,0,1,1,0},
                    {1,1,1,0,0,1,1,1},
                    {1,1,1,0,1,0,0,0},
                    {1,1,1,0,1,0,0,1},
                    {1,1,1,0,1,0,1,0},
                    {1,1,1,0,1,0,1,1},
                    {1,1,1,0,1,1,0,0},
                    {1,1,1,0,1,1,0,1},
                    {1,1,1,0,1,1,1,0},
                    {1,1,1,0,1,1,1,1},
                    {1,1,1,1,0,0,0,0},
                    {1,1,1,1,0,0,0,1},
                    {1,1,1,1,0,0,1,0},
                    {1,1,1,1,0,0,1,1},
                    {1,1,1,1,0,1,0,0},
                    {1,1,1,1,0,1,0,1},
                    {1,1,1,1,0,1,1,0},
                    {1,1,1,1,0,1,1,1},
                    {1,1,1,1,1,0,0,0},
                    {1,1,1,1,1,0,0,1},
                    {1,1,1,1,1,0,1,0},
                    {1,1,1,1,1,0,1,1},
                    {1,1,1,1,1,1,0,0},
                    {1,1,1,1,1,1,0,1},
                    {1,1,1,1,1,1,1,0},
                    {1,1,1,1,1,1,1,1}
            };

public:
    PacketFramer()
    : m_crc_enable(true)
    , m_pkt(nullptr)
    {
        m_pkt = new data_type[MAX_PKT_SIZE_BITS];

        memset(m_pkt, 0, MAX_PKT_SIZE_BITS); // clear buffer

        // Load the Unique word here
        for(size_t i = 0; i < UNIQUE_WORD_SIZE_BIT; i++)
        {
            m_pkt[i] = PACKET_FRAMER_UNIQUE_WORD[i];
        }
    }

    ~PacketFramer()
    {
        if(m_pkt)
        {
            delete[] m_pkt;
        }
    }

    void checkCRC(bool check)
    {
        m_crc_enable = check;
    }

    bool checkCRC() const
    {
        return m_crc_enable;
    }

    int serialize(uint8_t* data, size_t length, BBP_Block* pkt)
    {
        if(length > DATA_SIZE_BYTE)
        {
            return -1; // packet is to large
        }

        // flush previous packet data
        memset(&m_pkt[LENGTH_POS], 0, MAX_PKT_SIZE_BITS - 16);

        std::cout << "BANG" << std::endl;
        // load packet length
        for(int i = 0; i < 8; i++)
        {
            m_pkt[LENGTH_POS+i] = binary_array[length][i];
        }

        // crc insert
        if(m_crc_enable)
        {
            uint8_t crc = crcCalculator.getCRC(data, length);
            for(int i = 0; i < 8; i++)
            {
                m_pkt[CRC_POS + i] = binary_array[crc][i];
            }
        }
        else {
            // if crc is not enabled, load the crc slot with all zeros
            for (int i = 0; i < 8; i++) {
                m_pkt[CRC_POS + i] = 0;
            }
        }
        // load data
        for(size_t i = 0; i < length; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                m_pkt[DATA_POS+(i*8)+j] = binary_array[data[i]][j];
            }
        }

        // Load the BBP Block with packet -- Load a full packet
        for(size_t i = 0; i < MAX_PKT_SIZE_BITS; i++)
        {
            pkt->points[i].real(m_pkt[i]);
            pkt->points[i].imag(0);
            pkt->n_points++;
        }

        return MAX_PKT_SIZE_BITS; // total packet size in bits
    }

    int deserialize(RingBuffer<data_type>& buf, uint8_t* pkt, size_t pkt_buffer_size, bool& validPacket, bool& validCRC, bool& validHeader, int& bad_bits)
    {
        validPacket = false;
        validCRC = false;
        validHeader = false;
        bad_bits = 0;

        // Make sure we have a place to put the packet
        if(pkt_buffer_size < 250)
        {
            return 0; // Invalid dropbox
        }

        // Validate the total number of bits is a full packet
        if(buf.count() < MAX_PKT_SIZE_BITS) {
            return 0; // not enough stuff
        }

        // work until we find a header

        do
        {
            validHeader= true;
            // Find the Header
            for(size_t i = 0; i < UNIQUE_WORD_SIZE_BIT; i++)
            {
                if(buf.value(i) != m_pkt[i])
                {
                    bad_bits++;
                    buf.remove(1);
                    validHeader = false;
                    i = 0;

                    if(bad_bits >= 500) // allows for status updates
                    {
                        return 0;
                    }
                }
            }

            if(buf.count() < MAX_PKT_SIZE_BITS)
            {
                return 0;
            }
        }
        while(!validHeader);

        // IF WE ARE HERE WE HAVE A GOOD HEADER
        validHeader = true;

        uint8_t pkt_length = ToByte(buf, LENGTH_POS);

        if(pkt_length > 250)
        {
            bad_bits = 1;
            return -1;
        }

        // Got a good length here
        uint8_t pkt_crc = ToByte(buf, CRC_POS);
        memset(pkt, 0, MAX_BYTES_DATA); // clear the data dropbox

        for(size_t i = 0; i < MAX_BYTES_DATA; i++)
        {
            pkt[i] = ToByte(buf,i*8 + DATA_POS);
        }

        // check crc if enabled
        if(m_crc_enable)
        {
            if(crcCalculator.getCRC(pkt, pkt_length) == pkt_crc)
            {
                validCRC = true; // CRC is valid
            }
            else
            {
                bad_bits = 1;
                return -1; // Invalid CRC flag is already set
            }
        }
        else
        {
            // Always set true if we aren't checking
            validCRC = true;
        }

        // Data message is loaded and verified
        validPacket = true;
        return MAX_PKT_SIZE_BYTE; // Signal that we need to remove the whole packet from the buffer
    }

    static inline uint8_t ToByte(data_type* buffer, size_t index)
    {
        uint8_t c = 0;
        for (size_t i=0; i < 8; ++i)
        {
            if (buffer[index+i] == 1)
            {
                c |= 1 << (7-i);
            }
        }

        return c;
    }

    static inline uint8_t ToByte(RingBuffer<data_type>& buffer, size_t index)
    {
        uint8_t c = 0;
        for (size_t i=0; i < 8; ++i)
        {
            if (buffer.value(index+i) == 1)
            {
                c |= 1 << (7-i);
            }
        }

        return c;
    }
};


#endif //RADIONODE_PACKETFRAMER_H
