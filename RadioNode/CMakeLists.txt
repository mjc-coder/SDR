cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_helpers/")


SET(CMAKE_CXX_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
SET(CMAKE_C_FLAGS "-g -O0 -Wall -W -fprofile-arcs -ftest-coverage")
SET(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")

# Include Source Directories
include_directories(
        ${CMAKE_SOURCE_DIR}/src/application/
        ${CMAKE_SOURCE_DIR}/src/libraries/
        ${CMAKE_SOURCE_DIR}/src/test/
        ${CMAKE_SOURCE_DIR}/third-party/
        ${CMAKE_SOURCE_DIR}/third-party/hackrf/
        ${CMAKE_SOURCE_DIR}/third-party/hackrf/hackrf/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/include/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/convenience/
        ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/getopt/
        /opt/local/include/libhackrf/
        /usr/local/include)

# include targets
include(${CMAKE_SOURCE_DIR}/src/test/TESTS.cmake)
#include(${CMAKE_SOURCE_DIR}/src/application/hackrf_radio_node/hackrf.cmake)
#include(${CMAKE_SOURCE_DIR}/src/application/rtlsdr_radio_node/rtlsdr.cmake)




#[[


# Include Source
add_executable( rtl_fm  RTLSDR/rtl_fm.c
                RTLSDR/librtlsdr.c
                RTLSDR/tuner_e4k.c
                RTLSDR/tuner_fc0012.c
                RTLSDR/tuner_fc0013.c
                RTLSDR/tuner_fc2580.c
                RTLSDR/tuner_r82xx.c
                RTLSDR/getopt/getopt.c
                RTLSDR/convenience/convenience.c src/libraries/DigitalSignalProcessing/NotchFilter.cpp src/libraries/DigitalSignalProcessing/NotchFilter.h src/libraries/DigitalSignalProcessing/FFT.cpp src/libraries/DigitalSignalProcessing/FFT.h src/libraries/DigitalSignalProcessing/DC_Filter.cpp src/libraries/DigitalSignalProcessing/DC_Filter.h)

# Link system libraries
target_link_libraries(rtl_fm /usr/local/lib/libusb-1.0.dylib)



# Include Source
add_executable( am_demod_test
                RTLSDR/librtlsdr.c
                RTLSDR/tuner_e4k.c
                RTLSDR/tuner_fc0012.c
                RTLSDR/tuner_fc0013.c
                RTLSDR/tuner_fc2580.c
                RTLSDR/tuner_r82xx.c
                RTLSDR/getopt/getopt.c
                RTLSDR/convenience/convenience.c
        Scrap.cpp
        src/libraries/common/Baseband_Stream.cpp
        src/test/am_demod_test.cpp src/test/TEST_BPSK_Demodulator.cpp src/libraries/DigitalSignalProcessing/NotchFilter.cpp src/libraries/DigitalSignalProcessing/NotchFilter.h src/libraries/DigitalSignalProcessing/fifth_order_filter.cpp src/libraries/DigitalSignalProcessing/fifth_order_filter.h src/libraries/DigitalSignalProcessing/FFT.cpp src/libraries/DigitalSignalProcessing/FFT.h src/libraries/DigitalSignalProcessing/DC_Filter.cpp src/libraries/DigitalSignalProcessing/DC_Filter.h src/libraries/DigitalSignalProcessing/LowPassFilter.cpp src/libraries/DigitalSignalProcessing/LowPassFilter.h src/libraries/DigitalSignalProcessing/FirstOrderDigitalIntegration.cpp src/libraries/DigitalSignalProcessing/FirstOrderDigitalIntegration.h src/libraries/DigitalSignalProcessing/Normalize.cpp src/libraries/DigitalSignalProcessing/Normalize.h src/libraries/DigitalSignalProcessing/Resample.cpp src/libraries/DigitalSignalProcessing/Resample.h src/libraries/common/BBP_Block.cpp src/libraries/common/BBP_Block.h src/libraries/common/SafeBufferPoolQueue.cpp src/libraries/common/SafeBufferPoolQueue.h src/libraries/DigitalSignalProcessing/All_DigitalSignalProcessing.h src/libraries/common/All_Common.h src/libraries/streams/All_Streams.h)

# Link system libraries
target_link_libraries(am_demod_test Boost::program_options Boost::system /usr/local/lib/libusb-1.0.dylib)

# Include Source
add_executable( fm_demod_test
                RTLSDR/librtlsdr.c
                RTLSDR/tuner_e4k.c
                RTLSDR/tuner_fc0012.c
                RTLSDR/tuner_fc0013.c
                RTLSDR/tuner_fc2580.c
                RTLSDR/tuner_r82xx.c
                RTLSDR/getopt/getopt.c
                RTLSDR/convenience/convenience.c
        Scrap.cpp
        src/libraries/common/Baseband_Stream.cpp
        src/test/fm_demod_test.cpp src/libraries/DigitalSignalProcessing/NotchFilter.cpp src/libraries/DigitalSignalProcessing/NotchFilter.h src/libraries/DigitalSignalProcessing/FFT.cpp src/libraries/DigitalSignalProcessing/FFT.h src/libraries/DigitalSignalProcessing/DC_Filter.cpp src/libraries/DigitalSignalProcessing/DC_Filter.h)

# Link system libraries
target_link_libraries(fm_demod_test Boost::program_options Boost::system /usr/local/lib/libusb-1.0.dylib)


# Include Source
add_executable(hackrf_node
        Scrap.cpp
        third-party/FIR-filter-class/filt.cpp
        src/libraries/hackrf/HackRF_radio.cpp
        src/application/hackrf_radio_node/hackrf_main.cpp src/libraries/DigitalSignalProcessing/NotchFilter.cpp src/libraries/DigitalSignalProcessing/NotchFilter.h src/libraries/DigitalSignalProcessing/FFT.cpp src/libraries/DigitalSignalProcessing/FFT.h src/libraries/DigitalSignalProcessing/DC_Filter.cpp src/libraries/DigitalSignalProcessing/DC_Filter.h)

# Link system libraries
target_link_libraries(hackrf_node Boost::program_options Boost::system /usr/local/lib/libusb-1.0.dylib /opt/local/lib/libhackrf.dylib /usr/local/lib/libliquid.dylib)

]]
