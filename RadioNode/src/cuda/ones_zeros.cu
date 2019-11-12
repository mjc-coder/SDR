
#include <streams/AM.h>
#include <common/Common_Deffinitions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

extern "C"
struct am_demod_point
{
    const RADIO_DATA_TYPE m_threshold; ///< Decision Threshold

    /// Constructor
    am_demod_point(RADIO_DATA_TYPE threshold)
    : m_threshold(threshold)
    {
    }

    __host__ __device__
        RADIO_DATA_TYPE operator()(const RADIO_DATA_TYPE& x, const RADIO_DATA_TYPE& y) const {
            return (sqrt((x*x) + (y*y)) >= m_threshold) ? 1.0 : 0.0;
        }
};


extern "C"
int32_t ones_zeros_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t number_of_points)
{
   std::cout << "Ones/Zeros Demodulation is working" << std::endl;

    // Push data to the GPU for processing
    thrust::device_vector<RADIO_DATA_TYPE> d_vec_x(real, real+number_of_points);
    thrust::device_vector<RADIO_DATA_TYPE> d_vec_y(imag, imag+number_of_points);
    thrust::device_vector<uint8_t> d_output_vec(number_of_points);

    // AM Demodulate Data
    thrust::transform(d_vec_x.begin(), d_vec_x.end(), d_vec_y.begin(), d_vec_y.begin(), am_demod_point(AM<uint8_t>::THRESHOLD));
    cudaDeviceSynchronize(); // block until kernel is finished

    int32_t sum = thrust::reduce(d_output_vec.begin(), d_output_vec.end(), (int) 0, thrust::plus<int>());
    cudaDeviceSynchronize(); // block until kernel is finished


    if ( cudaSuccess != cudaGetLastError() )
    {
        std::cout << "[CUDA][am_gpu_demodulation] Error!" << std::endl;
    }

    return sum;
}
