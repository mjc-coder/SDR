#include <QCoreApplication>
#include <chrono>
#include <ratio>
#include <streams/AM.h>
#include <iostream>
#include <common/Common_Deffinitions.h>
#include <fstream>

#include <streams/FM.h>



#ifdef GPU_ENABLED

    extern "C"
    void am_gpu_initialize(size_t MAX_NUM_POINTS, size_t num_of_radios);

    extern "C"
    void am_gpu_free();

    extern "C"
    double am_gpu_demodulation(int RADIO_ID, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds);

    extern "C"
    void fm_gpu_free();

    extern "C"
    void fm_gpu_initialize(size_t num_of_radios, size_t MaxBufferSize, size_t rx_sample_rate, size_t samples_per_bit, double MARK_FREQ, double SPACE_FREQ);

    extern "C"
    double fm_gpu_demodulation(int RADIO_ID, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds);

    extern "C"
    int32_t ones_zeros_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t number_of_points);
#endif



#define MAX 200

int main(int argc, char *argv[])
{
    unsigned long long BufferSize = 512000;
    using namespace std::chrono;
    RADIO_DATA_TYPE* Buffer = new RADIO_DATA_TYPE[BufferSize*MAX];
    uint8_t* OUTPUT = new uint8_t[BufferSize*MAX];
    ::AM<RADIO_DATA_TYPE> am_demodulator;
    ::FM<RADIO_DATA_TYPE> fm_demodulator(10000000, 2000000);

    std::ofstream fout_cpu_am("CPU_AM.dat", std::ios::out | std::ios::trunc);
    std::ofstream fout_gpu_am("GPU_AM.dat", std::ios::out | std::ios::trunc);
    std::ofstream fout_cpu_fm("CPU_FM.dat", std::ios::out | std::ios::trunc);
    std::ofstream fout_gpu_fm("GPU_FM.dat", std::ios::out | std::ios::trunc);

    am_gpu_initialize(BufferSize*MAX, 1);
    fm_gpu_initialize(1, MAX*BufferSize, 2000000, 100, 10, 20);


    // Quick Qual check,  make sure each of the functions matches eachother
    {
        RADIO_DATA_TYPE* Buffer_input = new RADIO_DATA_TYPE[BufferSize*MAX];
        uint8_t* OUTPUT_CPU = new uint8_t[BufferSize];
        uint8_t* OUTPUT_GPU = new uint8_t[BufferSize];

        // initialize array
        for(size_t i = 0; i < BufferSize; i++)
        {
            Buffer_input[i] = i%255;
        }

        am_demodulator.demodulate(Buffer_input, Buffer_input, OUTPUT_CPU, BufferSize, 1);

        am_gpu_demodulation(0, Buffer_input, Buffer_input, OUTPUT_GPU, BufferSize, 1);

        for(size_t i = 0; i < BufferSize; i++)
        {
            if(OUTPUT_CPU[i] != OUTPUT_GPU[i])
            {
                std::cerr << "AM  " << i << "  " << OUTPUT_CPU[i] << "  " << OUTPUT_GPU[i] << std::endl;
            }
        }

        delete[] Buffer_input;
        delete[] OUTPUT_CPU;
        delete[] OUTPUT_GPU;
    }


    // Quick Qual check,  make sure each of the functions matches eachother
    {
        RADIO_DATA_TYPE* Buffer_input = new RADIO_DATA_TYPE[BufferSize*MAX];
        uint8_t* OUTPUT_CPU = new uint8_t[BufferSize];
        uint8_t* OUTPUT_GPU = new uint8_t[BufferSize];

        // initialize array
        for(size_t i = 0; i < BufferSize; i++)
        {
            Buffer_input[i] = i%255;
        }

        fm_demodulator.demodulate(Buffer_input, Buffer_input, OUTPUT_CPU, BufferSize, 1);

        fm_gpu_demodulation(0, Buffer_input, Buffer_input, OUTPUT_GPU, BufferSize, 1);

        for(size_t i = 0; i < BufferSize; i++)
        {
            if(OUTPUT_CPU[i] != OUTPUT_GPU[i])
            {
                std::cerr << "FM " << i << "  " << OUTPUT_CPU[i] << "  " << OUTPUT_GPU[i] << std::endl;
            }
        }

        delete[] Buffer_input;
        delete[] OUTPUT_CPU;
        delete[] OUTPUT_GPU;
    }



    // initialize Buffer array
    // initialize array
    for(size_t i = 0; i < BufferSize*MAX; i++)
    {
        Buffer[i] = i%255;
    }

    for(int i = 1; i <= MAX; i++)
    {
        std::cerr << "Demodulating [AM][CPU][" << i << "][" << BufferSize*i << "]";

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        am_demodulator.demodulate(Buffer, Buffer, OUTPUT, BufferSize*i, 1); // downsample of 1 is the most work

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cerr << " Total Time " << time_span.count() << std::endl;
        fout_cpu_am << i*BufferSize << ',' << time_span.count() << ',' << std::endl;
    }


    for(int i = 1; i <= MAX; i++)
    {
        std::cerr << "Demodulating [AM][GPU][" << i << "][" << BufferSize*i << "]";

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        double calc = am_gpu_demodulation(0, Buffer, Buffer, OUTPUT, BufferSize*i, 1); // downsample of 1 is the most work

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cerr << "[Calc Time " << calc << "] " << " Total Time " << time_span.count() << std::endl;
        fout_gpu_am << i*BufferSize << ',' << time_span.count() << ',' << calc << ',' << std::endl;
    }


    for(int i = 1; i <= MAX; i++)
    {
        std::cerr << "Demodulating [FM][CPU][" << i << "][" << BufferSize*i << "]";

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        fm_demodulator.demodulate(Buffer, Buffer, OUTPUT, BufferSize*i, 100); // downsample must be greater than 1 for FM.  1000 is pretty good

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cerr << " Total Time " << time_span.count() << std::endl;
        fout_cpu_fm << i*BufferSize << ',' << time_span.count() << ',' << std::endl;
    }



    for(int i = 1; i <= MAX; i++)
    {
        std::cerr << "Demodulating [FM][GPU][" << i << "][" << BufferSize*i << "]";

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        double calc = fm_gpu_demodulation(0, Buffer, Buffer, OUTPUT, BufferSize*i, 100); // downsample must be greater than 1 for FM.  1000 is pretty good

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cerr << "[Calc Time " << calc << "] " << " Total Time " << time_span.count() << std::endl;
        fout_gpu_am << i*BufferSize << ',' << time_span.count() << ',' << calc << ',' << std::endl;
    }

    am_gpu_free();
    fm_gpu_free();

    delete[] Buffer;
    delete[] OUTPUT;
}
