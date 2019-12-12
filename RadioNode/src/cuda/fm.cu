
#include <streams/AM.h>
#include <common/Common_Deffinitions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <chrono>


extern "C"
__global__ void kernel_fm_demod(RADIO_DATA_TYPE* op1, RADIO_DATA_TYPE* op2, RADIO_DATA_TYPE* Wave1, RADIO_DATA_TYPE* Wave2,
                                uint8_t* output, size_t opSize, size_t WaveSize, size_t ds)
{
    double op_one = 0;
    double op_two = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < opSize && i % ds == 0)
    {
        // convolve for the point we need
        for (size_t j = 0; j < WaveSize; j++)
        {
            op_one += op1[i + WaveSize - j] * Wave1[WaveSize - j - 1];
            op_two += op2[i + WaveSize - j] * Wave2[WaveSize - j - 1];
        }
        output[i/ds] = (op_one > op_two) ? 1 : 0;
    }
}


struct fm_gpu_variables
{
    RADIO_DATA_TYPE* host_WAVE1;
    RADIO_DATA_TYPE* host_WAVE2;
    uint8_t* dev_output;
    RADIO_DATA_TYPE* dev_Wave1;
    RADIO_DATA_TYPE* dev_Wave2;
    RADIO_DATA_TYPE* dev_op1;
    RADIO_DATA_TYPE* dev_op2;
    double MARK_FREQ;
    double SPACE_FREQ;
};

fm_gpu_variables* fmVariables;
size_t g_num_fm_gpu_variables;

extern "C"
void fm_gpu_initialize(size_t num_of_radios, size_t MaxBufferSize, size_t rx_sample_rate, size_t samples_per_bit, double MARK_FREQ, double SPACE_FREQ)
{
    fmVariables = new fm_gpu_variables[num_of_radios];
    g_num_fm_gpu_variables = num_of_radios;

    for(int i = 0; i < num_of_radios; i++)
    {
        fmVariables[i].MARK_FREQ = MARK_FREQ;
        fmVariables[i].SPACE_FREQ = SPACE_FREQ;
        // Build the wave once and share it between all of them
        fmVariables[i].host_WAVE1 = new RADIO_DATA_TYPE[samples_per_bit];
        fmVariables[i].host_WAVE2 = new RADIO_DATA_TYPE[samples_per_bit];

        const double delta_t = 1.0/rx_sample_rate;
        // Generate Mark and Space convolution waves
        for(size_t j = 0; j < samples_per_bit; j++)
        {
            fmVariables[i].host_WAVE1[j] = sqrt(2.0/samples_per_bit)*cos(2.0*PI*MARK_FREQ*j*delta_t);
            fmVariables[i].host_WAVE2[j] = sqrt(2.0/samples_per_bit)*cos(2.0*PI*SPACE_FREQ*j*delta_t);
        }

        // Initialize variables
        cudaMalloc((void**)&fmVariables[i].dev_output, MaxBufferSize*sizeof(uint8_t));
        cudaMalloc((void**)&fmVariables[i].dev_Wave1,  samples_per_bit*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&fmVariables[i].dev_Wave2,  samples_per_bit*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&fmVariables[i].dev_op1,    (MaxBufferSize+2*samples_per_bit)*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&fmVariables[i].dev_op2,    (MaxBufferSize+2*samples_per_bit)*sizeof(RADIO_DATA_TYPE));

        // Load memory to GPU
        cudaMemcpy(fmVariables[i].dev_Wave1, fmVariables[i].host_WAVE1, samples_per_bit*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(fmVariables[i].dev_Wave2, fmVariables[i].host_WAVE2, samples_per_bit*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    }
}

extern "C"
void fm_gpu_free()
{
    for(int i = 0; i < g_num_fm_gpu_variables; i++)
    {
        cudaFree(fmVariables[i].dev_output);
        cudaFree(fmVariables[i].dev_Wave1);
        cudaFree(fmVariables[i].dev_Wave2);
        cudaFree(fmVariables[i].dev_op1);
        cudaFree(fmVariables[i].dev_op2);
        delete[] fmVariables[i].host_WAVE1;
        delete[] fmVariables[i].host_WAVE2;
    }
}

extern "C"
void fm_gpu_set_samples_per_bit(size_t samples_per_bit, size_t rx_sample_rate)
{
    for(int i = 0; i < g_num_fm_gpu_variables; i++)
    {
        // initialize must be called first
        delete[] fmVariables[i].host_WAVE1;
        delete[] fmVariables[i].host_WAVE2;
        cudaFree(fmVariables[i].dev_Wave1);
        cudaFree(fmVariables[i].dev_Wave2);


        // Build the wave once and share it between all of them
        fmVariables[i].host_WAVE1 = new RADIO_DATA_TYPE[samples_per_bit];
        fmVariables[i].host_WAVE2 = new RADIO_DATA_TYPE[samples_per_bit];

        const double delta_t = 1.0/rx_sample_rate;
        // Generate Mark and Space convolution waves
        for(size_t j = 0; j < samples_per_bit; j++)
        {
            fmVariables[i].host_WAVE1[j] = sqrt(2.0/samples_per_bit)*cos(2.0*PI*fmVariables[i].MARK_FREQ*j*delta_t);
            fmVariables[i].host_WAVE2[j] = sqrt(2.0/samples_per_bit)*cos(2.0*PI*fmVariables[i].SPACE_FREQ*j*delta_t);
        }

        cudaMalloc((void**)&fmVariables[i].dev_Wave1,  samples_per_bit*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&fmVariables[i].dev_Wave2,  samples_per_bit*sizeof(RADIO_DATA_TYPE));

        // Load memory to GPU
        cudaMemcpy(fmVariables[i].dev_Wave1, fmVariables[i].host_WAVE1, samples_per_bit*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(fmVariables[i].dev_Wave2, fmVariables[i].host_WAVE2, samples_per_bit*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    }
}

extern "C"
double fm_gpu_demodulation(int RADIO_ID, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds)
{
    // Load data for convolution calculation
    cudaMemset(fmVariables[RADIO_ID].dev_op1, 0, (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));
    cudaMemset(fmVariables[RADIO_ID].dev_op2, 0, (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));
    cudaMemcpy(&fmVariables[RADIO_ID].dev_op1[ds+1], real, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(&fmVariables[RADIO_ID].dev_op2[ds+1], imag, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);


    // GPU Processing
    int threadsPerBlock = 1024;
    int blocksPerGrid =(number_of_points + threadsPerBlock - 1) / threadsPerBlock;

// FM Processing
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // call kernel
    kernel_fm_demod<<<blocksPerGrid, threadsPerBlock>>>( fmVariables[RADIO_ID].dev_op1, fmVariables[RADIO_ID].dev_op2, fmVariables[RADIO_ID].dev_Wave1, fmVariables[RADIO_ID].dev_Wave2, fmVariables[RADIO_ID].dev_output, number_of_points, ds, ds);
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
// FM Done

    // pull data back -- only pull back what we need #num points / ds
    cudaMemcpy(output, fmVariables[RADIO_ID].dev_output, number_of_points/ds, cudaMemcpyDeviceToHost);


    if ( cudaSuccess != cudaGetLastError() )
    {
        std::cout << "[CUDA][am_gpu_demodulation] Error!" << std::endl;
    }

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    return time_span.count();
}
