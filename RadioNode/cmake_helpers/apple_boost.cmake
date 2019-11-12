## Includes boost libraries on the apple machines


set(Boost_DEBUG 1)

# Include Boost
set(Boost_NO_SYSTEM_PATHS TRUE)
if (Boost_NO_SYSTEM_PATHS)
    set(BOOST_ROOT "/usr/local/Cellar/boost/1.70.0")
    set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
    set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
endif (Boost_NO_SYSTEM_PATHS)

find_package(Boost 1.70 COMPONENTS program_options REQUIRED)
find_package(Boost 1.70 COMPONENTS system REQUIRED)