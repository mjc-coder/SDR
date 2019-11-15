
#include <streams/AM.h>
#include <common/Common_Deffinitions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <chrono>

//extern "C"
//struct am_demod_point
//{
//    const RADIO_DATA_TYPE m_threshold; ///< Decision Threshold

//    /// Constructor
//    am_demod_point(RADIO_DATA_TYPE threshold)
//    : m_threshold(threshold)
//    {
//    }

//    __host__ __device__
//        RADIO_DATA_TYPE operator()(const RADIO_DATA_TYPE& x, const RADIO_DATA_TYPE& y) const {
//            return (sqrt((x*x) + (y*y)) >= m_threshold) ? 1.0 : 0.0;
//        }
//};

//extern "C"
//__global__ void downsample(RADIO_DATA_TYPE* d_vec_x, uint8_t* output, size_t N, size_t ds)
//{
//    int tid = blockIdx.x;
//    if(tid < N && tid % ds == 0)
//    {
//        output[tid/ds] = d_vec_x[tid];
//    }
//}

//extern "C"
//size_t am_gpu_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds)
//{
//    size_t N = number_of_points;
//    std::cout << "AM GPU Demodulation is working" << std::endl;

//    // Push data to the GPU for processing
//    thrust::device_vector<RADIO_DATA_TYPE> d_vec_x(real, real+number_of_points);
//    thrust::device_vector<RADIO_DATA_TYPE> d_vec_y(imag, imag+number_of_points);
//    thrust::device_vector<uint8_t> d_output_vec(number_of_points/ds);

//    // AM Demodulate Data
//    thrust::transform(d_vec_x.begin(), d_vec_x.end(), d_vec_y.begin(), d_vec_y.begin(), am_demod_point(AM<uint8_t>::THRESHOLD));
//    cudaDeviceSynchronize(); // block until kernel is finished

//    RADIO_DATA_TYPE* raw_input = thrust::raw_pointer_cast(d_vec_y.data()); // data returns to us in y not x
//    uint8_t* raw_output = thrust::raw_pointer_cast(d_output_vec.data());

//    // Downsample data on the device
//    downsample<<<N, 1>>>( raw_input, raw_output, N, ds );
//    cudaDeviceSynchronize(); // block until kernel is finished
//    // transfer data back to host
//    thrust::copy(d_output_vec.begin(), d_output_vec.end(), output);


//    if ( cudaSuccess != cudaGetLastError() )
//    {
//        std::cout << "[CUDA][am_gpu_demodulation] Error!" << std::endl;
//    }

//    return number_of_points/ds;
//}


extern "C"
__global__ void kernel_am_demod(RADIO_DATA_TYPE* x, RADIO_DATA_TYPE* y, uint8_t* output, size_t N, size_t ds)
{
    const double Threshold = 80;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && i % ds == 0)
    {
       output[i/ds] = (sqrt((x[i]*x[i]) + (y[i]*y[i])) >= Threshold) ? 1.0 : 0.0;
    }
}


struct am_gpu_variables
{
    RADIO_DATA_TYPE* dev_real;
    RADIO_DATA_TYPE* dev_imag;
    uint8_t* dev_output;
};

am_gpu_variables* am_gpu_vars;
size_t g_num_am_radios;

extern "C"
void am_gpu_initialize(size_t MAX_NUM_POINTS, size_t num_of_radios = 1)
{
    g_num_am_radios = num_of_radios;
    am_gpu_vars = new am_gpu_variables[num_of_radios];
    for(int i = 0; i < num_of_radios; i++)
    {
        cudaMalloc((void**)&am_gpu_vars[i].dev_real,   MAX_NUM_POINTS*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&am_gpu_vars[i].dev_imag,   MAX_NUM_POINTS*sizeof(RADIO_DATA_TYPE));
        cudaMalloc((void**)&am_gpu_vars[i].dev_output, MAX_NUM_POINTS*sizeof(uint8_t));
    }
}

extern "C"
void am_gpu_free()
{
    for(int i = 0; i < g_num_am_radios; i++)
    {
        cudaFree(am_gpu_vars[i].dev_real);
        cudaFree(am_gpu_vars[i].dev_imag);
        cudaFree(am_gpu_vars[i].dev_output);
    }
    delete[] am_gpu_vars;
}

extern "C"
double am_gpu_demodulation(int RADIO_ID, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds)
{
    // load variables into GPU
    cudaMemcpy(am_gpu_vars[RADIO_ID].dev_real, real, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(am_gpu_vars[RADIO_ID].dev_imag, imag, number_of_points*sizeof(RADIO_DATA_TYPE), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 1024;
    int blocksPerGrid =(number_of_points + threadsPerBlock - 1) / threadsPerBlock;


    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // call kernel
    kernel_am_demod<<<blocksPerGrid, threadsPerBlock>>>( am_gpu_vars[RADIO_ID].dev_real, am_gpu_vars[RADIO_ID].dev_imag, am_gpu_vars[RADIO_ID].dev_output, number_of_points, ds);
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();


    // only pull back what we need
    cudaMemcpy(output, am_gpu_vars[RADIO_ID].dev_output, number_of_points/ds, cudaMemcpyDeviceToHost);


    if ( cudaSuccess != cudaGetLastError() )
    {
        std::cout << "[CUDA][am_gpu_demodulation] Error!" << std::endl;
    }


    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    return time_span.count();
}

