//
// Created by Micheal Cowan on 2019-07-21.
//



#include <iostream>
#include "Scrap.h"
#include "common/Baseband_Stream.h"
#include <fstream>
#include <src/streams/FM_Demodulator.h>


using namespace std;

void rotate_90(unsigned char *buf, uint32_t len)
/* 90 rotation is 1+0j, 0+1j, -1+0j, 0-1j
   or [0, 1, -3, 2, -4, -5, 7, -6] */
{
    uint32_t i;
    unsigned char tmp;
    for (i=0; i<len; i+=8) {
        /* uint8_t negation = 255 - x */
        tmp = 255 - buf[i+3];
        buf[i+3] = buf[i+2];
        buf[i+2] = tmp;

        buf[i+4] = 255 - buf[i+4];
        buf[i+5] = 255 - buf[i+5];

        tmp = 255 - buf[i+6];
        buf[i+6] = buf[i+7];
        buf[i+7] = tmp;
    }
}

int main()
{
    ifstream fin("../Samples/fm963_s2048000_100s.dat", std::ios::binary | std::ios::in);
    ofstream fout("../Samples/fm963_s32000_100s.pcm", std::ios::binary | std::ios::out | std::ios::trunc);

    BBP_Block block;
    Complex p0;
    Complex p1;

    while(!fin.eof())
    {
        block.reset();


        size_t m = 0;
        uint8_t ch[BLOCK_READ_SIZE*2] = {0};
        for(m = 0; fin.is_open() && m < BLOCK_READ_SIZE*2 && !fin.eof(); m++)
        {
            fin.read(reinterpret_cast<char *>(&ch[m]), 1);
        }

        for(size_t j = 0, n = 0; n < m; j++, n+=2)
        {
            block.points[j].real(ch[n] - 127.4);
            block.points[j].imag(ch[n+1] - 127.4);
        }

        block.n_points = m/2;

        fm_demod(&block, fout, &p0, &p1);
    }


    return 0;
}