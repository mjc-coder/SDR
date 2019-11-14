
#include <streams/AM.h>
#include <common/Common_Deffinitions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <chrono>


//// INPUT 1 - Read Data
//// INPUT 2 - Pulse SHAPE
//void convolve(const data_type* input1, size_t length1, const data_type* input2, size_t length2, data_type* output, size_t lengthOutput)
//{
//    RADIO_DATA_TYPE* real_data = new RADIO_DATA_TYPE[length1+(length2*2)];  // Upsample 8, and include forward buffer for convolution

//    memset(real_data, 0, (length1+(length2*2))*sizeof(RADIO_DATA_TYPE));

//    // Initialize data
//    for(size_t i = 0; i < length1; i++)
//    {
//        real_data[i+length2] = input1[i];
//    }

//    for(size_t i = 0; i < length1; i++) {
//        for (size_t j = 0; j < length2; j++) {
//            output[i] += real_data[i + length2 - j] * input2[length2 - j - 1];
//        }
//    }

//    delete[] real_data;
//}


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
        }

        // convolve for the point we need
        for (size_t j = 0; j < WaveSize; j++)
        {
            op_two += op2[i + WaveSize - j] * Wave2[WaveSize - j - 1];
        }
        output[i/ds] = (op_one > op_two) ? 1 : 0;
    }
}

extern "C"
double fm_gpu_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds, double rx_sample_rate, double MARK_FREQ, double SPACE_FREQ)
{
    uint8_t* dev_output;
    RADIO_DATA_TYPE* dev_Wave1;
    RADIO_DATA_TYPE* dev_Wave2;
    RADIO_DATA_TYPE* dev_op1;
    RADIO_DATA_TYPE* dev_op2;


    // Build the wave once and share it between all of them
    RADIO_DATA_TYPE* host_WAVE1 = new RADIO_DATA_TYPE[ds];
    RADIO_DATA_TYPE* host_WAVE2 = new RADIO_DATA_TYPE[ds];

    const double delta_t = 1.0/rx_sample_rate;
    // Generate Mark and Space convolution waves
    for(size_t i = 0; i < ds; i++)
    {
        host_WAVE1[i] = sqrt(2.0/ds)*cos(2.0*PI*MARK_FREQ*i*delta_t);
        host_WAVE2[i] = sqrt(2.0/ds)*cos(2.0*PI*SPACE_FREQ*i*delta_t);
    }


    cudaMalloc((void**)&dev_output, number_of_points*sizeof(uint8_t));
    cudaMalloc((void**)&dev_Wave1,  ds*sizeof(RADIO_DATA_TYPE));
    cudaMalloc((void**)&dev_Wave2,  ds*sizeof(RADIO_DATA_TYPE));
    cudaMalloc((void**)&dev_op1,    (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));
    cudaMalloc((void**)&dev_op2,    (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));

    // Load memory to GPU
    cudaMemcpy(dev_Wave1, host_WAVE1, ds*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Wave2, host_WAVE2, ds*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);

    // Load data for convolution calculation
    cudaMemset(dev_op1, 0, (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));
    cudaMemset(dev_op2, 0, (number_of_points+2*ds)*sizeof(RADIO_DATA_TYPE));
    cudaMemcpy(&dev_op1[ds+1], real, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_op2[ds+1], imag, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);


    // GPU Processing
    int threadsPerBlock = 1024;
    int blocksPerGrid =(number_of_points + threadsPerBlock - 1) / threadsPerBlock;

// FM Processing
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // call kernel
    kernel_fm_demod<<<blocksPerGrid, threadsPerBlock>>>( dev_op1, dev_op2, dev_Wave1, dev_Wave2, dev_output, number_of_points, ds, ds);
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
// FM Done

    cudaMemcpy(output, dev_output, number_of_points*sizeof(uint8_t), cudaMemcpyDeviceToHost); // pull data back


    if ( cudaSuccess != cudaGetLastError() )
    {
        std::cout << "[CUDA][am_gpu_demodulation] Error!" << std::endl;
    }

    cudaFree(dev_output);
    cudaFree(dev_Wave1);
    cudaFree(dev_Wave2);
    cudaFree(dev_op1);
    cudaFree(dev_op2);
    delete[] host_WAVE1;
    delete[] host_WAVE2;

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    return time_span.count();
}
