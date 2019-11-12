//
// Created by Micheal Cowan on 9/27/19.
//


#include <network/PacketFramer.h>
#include <gtest/gtest.h> // googletest header file


TEST(PacketFramer, CheckConstants)
{
    ASSERT_EQ(PacketFramer<uint8_t>::MAX_BYTES_DATA, 250);
    ASSERT_EQ(PacketFramer<uint8_t>::UNIQUE_WORD_SIZE_BIT, 16);
    ASSERT_EQ(PacketFramer<uint8_t>::UNIQUE_WORD_SIZE_BYTE, 2);
    ASSERT_EQ(PacketFramer<uint8_t>::UNIQUE_WORD_POS, 0);
    ASSERT_EQ(PacketFramer<uint8_t>::LENGTH_SIZE_BIT, 8);
    ASSERT_EQ(PacketFramer<uint8_t>::LENGTH_SIZE_BYTE, 1);
    ASSERT_EQ(PacketFramer<uint8_t>::LENGTH_POS, 16);
    ASSERT_EQ(PacketFramer<uint8_t>::CRC_SIZE_BIT, 8);
    ASSERT_EQ(PacketFramer<uint8_t>::CRC_SIZE_BYTE, 1);
    ASSERT_EQ(PacketFramer<uint8_t>::CRC_POS, 24);
    ASSERT_EQ(PacketFramer<uint8_t>::DATA_SIZE_BIT, 250*8);
    ASSERT_EQ(PacketFramer<uint8_t>::DATA_SIZE_BYTE, 250);
    ASSERT_EQ(PacketFramer<uint8_t>::DATA_POS, 32);
    ASSERT_EQ(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS, (250+4)*8);
    ASSERT_EQ(PacketFramer<uint8_t>::MAX_PKT_SIZE_BYTE, 250+4);
}


TEST(PacketFramer, FramedMessage)
{
    uint8_t message[] = {0x83, 0x01, 0x00, 0xAB};
    uint8_t FramedMessage[PacketFramer<uint8_t>::MAX_BYTES_DATA] = {1,1,1,1,1,0,1,0,    // UW 0
                                                                    1,1,0,0,1,1,1,0,    // UW 1
                                                                    0,0,0,0,0,1,0,0,    // LENGTH
                                                                    0,0,0,0,0,0,0,0,    // CRC
                                                                    1,0,0,0,0,0,1,1,    // DATA 0
                                                                    0,0,0,0,0,0,0,1,    // DATA 1
                                                                    0,0,0,0,0,0,0,0,    // DATA 2
                                                                    1,0,1,0,1,0,1,1};   // DATA 3


    PacketFramer<uint8_t> framer;
    framer.checkCRC(false);
    BBP_Block block(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS);
    framer.serialize(message, 4, &block);

    for(size_t i = 0; i < 8*10; i++) // overshoot and verify that we get all zeros
    {
        ASSERT_EQ(FramedMessage[i], block.points[i].real()) << "Failed at index " << i << "  " << (int)FramedMessage[i] << "  " << (int)block.points[i].real();
    }


};


TEST(PacketFramer, GoodDeframedPacket)
{
    uint8_t message[] = {0x83, 0x01, 0x00, 0xAB};
    uint8_t FramedMessage[PacketFramer<uint8_t>::MAX_BYTES_DATA] = {1,1,1,1,1,0,1,0,    // UW 0
                                                                    1,1,0,0,1,1,1,0,    // UW 1
                                                                    0,0,0,0,0,1,0,0,    // LENGTH
                                                                    0,0,0,0,0,0,0,0,    // CRC
                                                                    1,0,0,0,0,0,1,1,    // DATA 0
                                                                    0,0,0,0,0,0,0,1,    // DATA 1
                                                                    0,0,0,0,0,0,0,0,    // DATA 2
                                                                    1,0,1,0,1,0,1,1};   // DATA 3


    PacketFramer<uint8_t> framer;
    framer.checkCRC(false);
    uint8_t DeframedMessage[250] = {0};

    RingBuffer<uint8_t> buf(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS);
    bool validPacket;
    bool validCRC;
    bool validHeader;
    int bad_bits;
    int returnVal;


    struct solution
    {
        int returnVal;
        bool validPacket;
        bool validCRC;
        bool validHeader;
    };

    solution solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS];
    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1; i++)
    {
        solutionArray[i].returnVal = 0; // not enough bits
        solutionArray[i].validCRC = false;
        solutionArray[i].validHeader = false;
        solutionArray[i].validPacket = false;
    }
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].returnVal = PacketFramer<uint8_t>::MAX_PKT_SIZE_BYTE; // not enough bits
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validCRC = true;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validHeader = true;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validPacket = true;

    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS; i++) // overshoot and verify that we get all zeros
    {
        if(i < 64) {
            buf.append(FramedMessage[i]);
        }
        else
        {
            buf.append(0);
        }
        returnVal = framer.deserialize(buf, DeframedMessage, 250, validPacket, validCRC, validHeader, bad_bits);

        ASSERT_EQ(returnVal, solutionArray[i].returnVal) << "Return Failed at Index " << i << "  " << (int)returnVal << "  " << (int)solutionArray[i].returnVal << std::endl;
        ASSERT_EQ(validCRC, solutionArray[i].validCRC)  << "valid crc Failed at Index " << i;
        ASSERT_EQ(validHeader, solutionArray[i].validHeader)  << "valid header Failed at Index " << i;
        ASSERT_EQ(validPacket, solutionArray[i].validPacket)  << "valid packet Failed at Index " << i;
        ASSERT_EQ(bad_bits, 0)  << "bad bits Failed at Index " << i;
    }

    for(size_t i = 0; i < 4; i++)
    {
        ASSERT_EQ(message[i], DeframedMessage[i]) << "Failed at Index " << i << "  " << (int)message[i] << "   " << (int)DeframedMessage[i];
    }
};


TEST(PacketFramer, BadCRC)
{
    uint8_t message[] = {0x83, 0x01, 0x00, 0xAB};
    uint8_t FramedMessage[PacketFramer<uint8_t>::MAX_BYTES_DATA] = {1,1,1,1,1,0,1,0,    // UW 0
                                                                    1,1,0,0,1,1,1,0,    // UW 1
                                                                    0,0,0,0,0,1,0,0,    // LENGTH
                                                                    0,0,0,0,0,0,0,0,    // CRC
                                                                    1,0,0,0,0,0,1,1,    // DATA 0
                                                                    0,0,0,0,0,0,0,1,    // DATA 1
                                                                    0,0,0,0,0,0,0,0,    // DATA 2
                                                                    1,0,1,0,1,0,1,1};   // DATA 3


    PacketFramer<uint8_t> framer;
    framer.checkCRC(true);
    uint8_t DeframedMessage[250] = {0};

    RingBuffer<uint8_t> buf(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS);
    bool validPacket;
    bool validCRC;
    bool validHeader;
    int bad_bits;
    int returnVal;


    struct solution
    {
        int returnVal;
        bool validPacket;
        bool validCRC;
        bool validHeader;
        int badbits;
    };

    solution solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS];
    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1; i++)
    {
        solutionArray[i].returnVal = 0; // not enough bits
        solutionArray[i].validCRC = false;
        solutionArray[i].validHeader = false;
        solutionArray[i].validPacket = false;
        solutionArray[i].badbits=0;
    }
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].returnVal = -1; // not enough bits
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validCRC = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validHeader = true;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validPacket = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].badbits = 1;

    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS; i++) // overshoot and verify that we get all zeros
    {
        if(i < 64) {
            buf.append(FramedMessage[i]);
        }
        else
        {
            buf.append(0);
        }
        returnVal = framer.deserialize(buf, DeframedMessage, 250, validPacket, validCRC, validHeader, bad_bits);

        ASSERT_EQ(returnVal, solutionArray[i].returnVal) << "Return Failed at Index " << i << "  " << (int)returnVal << "  " << (int)solutionArray[i].returnVal << std::endl;
        ASSERT_EQ(validCRC, solutionArray[i].validCRC)  << "valid crc Failed at Index " << i;
        ASSERT_EQ(validHeader, solutionArray[i].validHeader)  << "valid header Failed at Index " << i;
        ASSERT_EQ(validPacket, solutionArray[i].validPacket)  << "valid packet Failed at Index " << i;
        ASSERT_EQ(bad_bits, solutionArray[i].badbits)  << "bad bits Failed at Index " << i;
    }

};


TEST(PacketFramer, BadHeader)
{
    uint8_t message[] = {0x83, 0x01, 0x00, 0xAB};
    uint8_t FramedMessage[PacketFramer<uint8_t>::MAX_BYTES_DATA] = {0,0,1,1,1,0,1,0,    // UW 0
                                                                    1,1,0,0,1,1,1,0,    // UW 1
                                                                    0,0,0,0,0,1,0,0,    // LENGTH
                                                                    0,0,0,0,0,0,0,0,    // CRC
                                                                    1,0,0,0,0,0,1,1,    // DATA 0
                                                                    0,0,0,0,0,0,0,1,    // DATA 1
                                                                    0,0,0,0,0,0,0,0,    // DATA 2
                                                                    1,0,1,0,1,0,1,1};   // DATA 3


    PacketFramer<uint8_t> framer;
    framer.checkCRC(true);
    uint8_t DeframedMessage[250] = {0};

    RingBuffer<uint8_t> buf(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS);
    bool validPacket;
    bool validCRC;
    bool validHeader;
    int bad_bits;
    int returnVal;


    struct solution
    {
        int returnVal;
        bool validPacket;
        bool validCRC;
        bool validHeader;
        int badbits;
    };

    solution solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS];
    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1; i++)
    {
        solutionArray[i].returnVal = 0; // not enough bits
        solutionArray[i].validCRC = false;
        solutionArray[i].validHeader = false;
        solutionArray[i].validPacket = false;
        solutionArray[i].badbits = 0;
    }
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].returnVal = 0; // not enough bits
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validCRC = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validHeader = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validPacket = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].badbits = 500;

    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS; i++) // overshoot and verify that we get all zeros
    {
        if(i < 64) {
            buf.append(FramedMessage[i]);
        }
        else
        {
            buf.append(0);
        }
        returnVal = framer.deserialize(buf, DeframedMessage, 250, validPacket, validCRC, validHeader, bad_bits);

        ASSERT_EQ(returnVal, solutionArray[i].returnVal) << "Return Failed at Index " << i << "  " << (int)returnVal << "  " << (int)solutionArray[i].returnVal << std::endl;
        ASSERT_EQ(validCRC, solutionArray[i].validCRC)  << "valid crc Failed at Index " << i;
        ASSERT_EQ(validHeader, solutionArray[i].validHeader)  << "valid header Failed at Index " << i;
        ASSERT_EQ(validPacket, solutionArray[i].validPacket)  << "valid packet Failed at Index " << i;
        ASSERT_EQ(bad_bits, solutionArray[i].badbits)  << "bad bits Failed at Index " << i;
    }

};


TEST(PacketFramer, BadLength)
{
    uint8_t message[] = {0x83, 0x01, 0x00, 0xAB};
    uint8_t FramedMessage[PacketFramer<uint8_t>::MAX_BYTES_DATA] = {1,1,1,1,1,0,1,0,    // UW 0
                                                                    1,1,0,0,1,1,1,0,    // UW 1
                                                                    1,1,1,1,1,1,1,1,    // LENGTH
                                                                    0,0,0,0,0,0,0,0,    // CRC
                                                                    1,0,0,0,0,0,1,1,    // DATA 0
                                                                    0,0,0,0,0,0,0,1,    // DATA 1
                                                                    0,0,0,0,0,0,0,0,    // DATA 2
                                                                    1,0,1,0,1,0,1,1};   // DATA 3


    PacketFramer<uint8_t> framer;
    framer.checkCRC(false);
    uint8_t DeframedMessage[250] = {0};

    RingBuffer<uint8_t> buf(PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS);
    bool validPacket;
    bool validCRC;
    bool validHeader;
    int bad_bits;
    int returnVal;


    struct solution
    {
        int returnVal;
        bool validPacket;
        bool validCRC;
        bool validHeader;
        int badbits;
    };

    solution solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS];
    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1; i++)
    {
        solutionArray[i].returnVal = 0; // not enough bits
        solutionArray[i].validCRC = false;
        solutionArray[i].validHeader = false;
        solutionArray[i].validPacket = false;
        solutionArray[i].badbits = 0;
    }
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].returnVal = -1; // not enough bits
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validCRC = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validHeader = true;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].validPacket = false;
    solutionArray[PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS-1].badbits = 1;

    for(size_t i = 0; i < PacketFramer<uint8_t>::MAX_PKT_SIZE_BITS; i++) // overshoot and verify that we get all zeros
    {
        if(i < 64) {
            buf.append(FramedMessage[i]);
        }
        else
        {
            buf.append(0);
        }
        returnVal = framer.deserialize(buf, DeframedMessage, 250, validPacket, validCRC, validHeader, bad_bits);

        ASSERT_EQ(returnVal, solutionArray[i].returnVal) << "Return Failed at Index " << i << "  " << (int)returnVal << "  " << (int)solutionArray[i].returnVal << std::endl;
        ASSERT_EQ(validCRC, solutionArray[i].validCRC)  << "valid crc Failed at Index " << i;
        ASSERT_EQ(validHeader, solutionArray[i].validHeader)  << "valid header Failed at Index " << i;
        ASSERT_EQ(validPacket, solutionArray[i].validPacket)  << "valid packet Failed at Index " << i;
        ASSERT_EQ(bad_bits, solutionArray[i].badbits)  << "bad bits Failed at Index " << i;
    }
};
