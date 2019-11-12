//
// Created by Micheal Cowan on 9/14/19.
//

#include <DigitalSignalProcessing/Resample.h>
#include <gtest/gtest.h> // googletest header file

TEST(Resample, upsample_with)
{
    RADIO_DATA_TYPE message[] = {-1,-1,-1,-1,1,1,1,1};
    RADIO_DATA_TYPE message_processed[64] = {0};
    const RADIO_DATA_TYPE final_message[64] = {-1,0,0,0,0,0,0,0,
                                               -1,0,0,0,0,0,0,0,
                                               -1,0,0,0,0,0,0,0,
                                               -1,0,0,0,0,0,0,0,
                                                1,0,0,0,0,0,0,0,
                                                1,0,0,0,0,0,0,0,
                                                1,0,0,0,0,0,0,0,
                                                1,0,0,0,0,0,0,0};


    upsample_fill_w_zeros(message, 8, message_processed, 64, 8);

    for(int i = 0; i < 64; i++)
    {
        ASSERT_EQ(final_message[i], message_processed[i]) << "Failed at index " << i;
    }
}

