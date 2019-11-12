//
// Created by Micheal Cowan on 2019-07-21.
//



#include <iostream>
#include "Scrap.h"
#include <libraries/common/All_Common.h>
#include <fstream>



using namespace std;

int main()
{
    ifstream fin("../Samples/gqrx_20190722_012714_126277000_240000_fc.raw", std::ios::binary | std::ios::in);
    ofstream fout("../Samples/gqrx_20190722_012714_126277000_240000_fc.pcm", std::ios::binary | std::ios::out | std::ios::trunc);

    BBP_Block block;
    // Low Pass filter
    LowPassFilter lpf_real(15000.0, 1.0/static_cast<RADIO_DATA_TYPE>(240000));


    while(!fin.eof()) {
        for (int i = 0; fin.is_open() && !fin.eof() && i < BLOCK_READ_SIZE; i++) {
            RADIO_DATA_TYPE ch = 0;
            fin.read(reinterpret_cast<char*>(&ch), 4);
            block.points[i].real(ch);
            fin.read(reinterpret_cast<char*>(&ch), 4);
            block.points[i].imag(ch);
            block.n_points++;
        }

        am_demod(block, fout, &lpf_real);
        block.reset();
    }


    return 0;
}