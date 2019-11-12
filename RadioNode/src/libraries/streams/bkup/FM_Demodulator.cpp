//
// Created by Micheal Cowan on 2019-08-01.
//

#include <streams/FM_Demodulator.h>
#include <FIR-filter-class/filt.h>
#include <DigitalSignalProcessing/fifth_order_filter.h>
#include <DigitalSignalProcessing/Normalize.h>
#include <DigitalSignalProcessing/Resample.h>

FM_Demodulator::FM_Demodulator( SafeBufferPoolQueue* BBP_Buffer_Pool,
                                std::string name,
                                std::string address,
                                std::string m_td_port,
                                std::string m_fd_port,
                                size_t array_size)
: Baseband_Stream([this](BBP_Block* block)
                    {
                        //demod(block);
                        demod2(block);
                    }, BBP_Buffer_Pool)
, m_dataoutput(BBP_Buffer_Pool, name, address, m_td_port, m_fd_port, array_size)
, quad(0)
, quad_delay_1(0)
, quad_delay_2(0)
, inphase(0)
, inphase_delay_1(0)
, inphase_delay_2(0)
, quad_prime(0)
, imag_prime(0)
, m_lpf(20000, 1.0/256000.0)
, m_notch(0.0033, 64500000)
, normalization_max(0)
{
    this->add_next_buffer(&m_dataoutput);
}

FM_Demodulator::~FM_Demodulator()
{
}

void FM_Demodulator::demod(BBP_Block* block)
{
    int num_points = 0;
    fifth_order_filter m_filter;
    // Downsample by 4
    m_filter.decimate(block, 4); // 1024000 => 256000

    // Point 0
    quad = block->points[0].imag();
    quad_delay_1 = p1.imag();
    quad_delay_2 = p0.imag();
    inphase = block->points[0].real();
    inphase_delay_1 = p1.real();
    inphase_delay_2 = p0.real();

    quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    m_ang[0] = quad_prime - imag_prime;

    // Point 1
    quad = block->points[1].imag();
    quad_delay_1 = block->points[0].imag();
    quad_delay_2 = p1.imag();
    inphase = block->points[1].real();
    inphase_delay_1 = block->points[0].real();
    inphase_delay_2 = p1.real();
    quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    m_ang[1] = quad_prime - imag_prime;

    // Loop through remaining
    for(size_t index = 2; index < block->number_of_points(); index++)
    {
        quad = block->points[index].imag();
        quad_delay_1 = block->points[index-1].imag();
        quad_delay_2 = block->points[index-2].imag();
        inphase = block->points[index].real();
        inphase_delay_1 = block->points[index-1].real();
        inphase_delay_2 = block->points[index-2].real();

        quad_prime = (quad - quad_delay_2) * inphase_delay_1;
        imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

        m_ang[index] = quad_prime - imag_prime;
    }

    // Store history
    p0.real(block->points[block->number_of_points()-2].real());
    p0.imag(block->points[block->number_of_points()-2].imag());
    p1.real(block->points[block->number_of_points()-1].real());
    p1.imag(block->points[block->number_of_points()-1].imag());


    m_notch.filter(m_ang, block->number_of_points());

    for(size_t i = 0; i < block->number_of_points(); i++)
    {
        m_ang[i] = m_lpf.update(m_ang[i]);
    }

    num_points = m_filter.decimate(m_ang, block->number_of_points(), 8); // 256000 => 32000
    normalization_max = normalize(m_ang, num_points, normalization_max);

    // write normalized points back to block array
    block->reset();
    for(int i = 0; i < num_points; i++)
    {
        block->points[i].real(m_ang[i]);
        block->points[i].imag(0);
    }
    block->n_points = num_points;
}



void FM_Demodulator::demod2(BBP_Block* block)
{
    int num_points = 0;

    // Point 0
    quad = block->points[0].imag();
    quad_delay_1 = p1.imag();
    quad_delay_2 = p0.imag();
    inphase = block->points[0].real();
    inphase_delay_1 = p1.real();
    inphase_delay_2 = p0.real();

    quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    m_ang[0] = quad_prime - imag_prime;

    // Point 1
    quad = block->points[1].imag();
    quad_delay_1 = block->points[0].imag();
    quad_delay_2 = p1.imag();
    inphase = block->points[1].real();
    inphase_delay_1 = block->points[0].real();
    inphase_delay_2 = p1.real();
    quad_prime = (quad - quad_delay_2) * inphase_delay_1;
    imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

    m_ang[1] = quad_prime - imag_prime;

    // Loop through remaining
    for(size_t index = 2; index < block->number_of_points(); index++)
    {
        quad = block->points[index].imag();
        quad_delay_1 = block->points[index-1].imag();
        quad_delay_2 = block->points[index-2].imag();
        inphase = block->points[index].real();
        inphase_delay_1 = block->points[index-1].real();
        inphase_delay_2 = block->points[index-2].real();

        quad_prime = (quad - quad_delay_2) * inphase_delay_1;
        imag_prime = (inphase - inphase_delay_2) * quad_delay_1;

        m_ang[index] = quad_prime - imag_prime;
    }

    // Store history
    p0.real(block->points[block->number_of_points()-2].real());
    p0.imag(block->points[block->number_of_points()-2].imag());
    p1.real(block->points[block->number_of_points()-1].real());
    p1.imag(block->points[block->number_of_points()-1].imag());

    // Downsample by 5
    Filter myFilter = Filter(LPF, 51, 240.000, 10.0);
    for(size_t i = 0; i < block->number_of_points(); i++)
    {
        m_ang[i] = myFilter.do_sample(m_ang[i]);
    }

    num_points = decimate(m_ang, block->number_of_points(), 5); // 240000 / 5 => 48000

    Filter myFilter2 = Filter(LPF, 31, 240.000/5.0, 3.0);
    for(size_t i = 0; i < num_points; i++)
    {
        m_ang[i] = myFilter2.do_sample(m_ang[i]);
    }

    normalization_max = normalize(m_ang, num_points, normalization_max);

    // write normalized points back to block array
    block->reset();
    for(int i = 0; i < num_points; i++)
    {
        block->points[i].real(m_ang[i]);
        block->points[i].imag(0);
    }
    block->n_points = num_points;
}