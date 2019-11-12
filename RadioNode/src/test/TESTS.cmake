cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)


# Includes
include(${CMAKE_SOURCE_DIR}/cmake_helpers/apple_gtest.cmake)
include(gatherSource)

gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/common SRC_COMMON)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/streams SRC_STREAMS)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/DigitalSignalProcessing SRC_DSP)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/network SRC_NETWORK)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/crc SRC_CRC)
gatherLibrarySource(${CMAKE_SOURCE_DIR}/src/libraries/arduino_if SRC_ARDUINO_IF)
gatherThirdPartySource(${CMAKE_SOURCE_DIR}/third-party/FIR-filter-class SRC_TP_FIR_FILTER)

## Add Unit Tests
add_executable(Radio_Unit_Tests
        ${SRC_COMMON}
        ${SRC_STREAMS}
        ${SRC_DSP}
        ${SRC_NETWORK}
        ${SRC_TP_FIR_FILTER}
        ${SRC_CRC}
        ${SRC_ARDUINO}
        src/test/TEST_PacketFramer.cpp
        src/test/TEST_AM.cpp
        src/test/TEST_FM.cpp
        src/test/TEST_BPSK.cpp
        src/test/TEST_QPSK.cpp
        src/test/TEST_crc.cpp
        src/test/TEST_Modulator_Demodulator.cpp
        src/test/TEST_Resample.cpp
        src/test/TEST_RingBuffer.cpp)

target_link_libraries(Radio_Unit_Tests ${GTEST_BOTH_LIBRARIES})

