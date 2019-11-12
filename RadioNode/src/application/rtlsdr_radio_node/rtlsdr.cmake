cmake_minimum_required(VERSION 3.14)

set(SOURCE "")

# Includes
include(gatherSource)
include(apple_boost)

gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/common COMMON_SRC)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/DigitalSignalProcessing DSP_SRC)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/network NETWORK_SRC)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/rtl_sdr RTL_SDR_SRC)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/streams STREAM_SRC)
gatherThirdPartySource(${CMAKE_SOURCE_DIR}/third-party/FIR-filter-class SRC_TP_FIR_FILTER)
gatherThirdPartySource(${CMAKE_SOURCE_DIR}/third-party/RTLSDR/convenience SRC_TP_RTLSDR_convenience)
set(SRC_TP_RTLSDR ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/librtlsdr.c
                  ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/tuner_e4k.c
                  ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/tuner_fc0012.c
                  ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/tuner_fc0013.c
                  ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/tuner_fc2580.c
                  ${CMAKE_SOURCE_DIR}/third-party/RTLSDR/tuner_r82xx.c)


message(">> >> ${TARGET_ID} >> ${SRC}")

## Add Unit Tests
add_executable(rtlsdr_radio_node
        ${CMAKE_SOURCE_DIR}/src/application/rtlsdr_radio_node/rtlsdr_main.cpp
        ${COMMON_SRC}
        ${DSP_SRC}
        ${NETWORK_SRC}
        ${RTL_SDR_SRC}
        ${STREAM_SRC}
        ${SRC_TP_RTLSDR}
        ${SRC_TP_FIR_FILTER}
        ${SRC_TP_RTLSDR_convenience})

target_link_libraries(rtlsdr_radio_node
        Boost::program_options
        Boost::system
        /usr/local/lib/libusb-1.0.dylib)