/// @file SDR/RadioNode/src/libraries/DigitalSignalProcessing/fifth_order_filter.h

#ifndef RADIONODE_FIFTH_ORDER_FILTER_H
#define RADIONODE_FIFTH_ORDER_FILTER_H

#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <stdint.h>

/// \brief Fifth Order decimation and Filter IIR
class fifth_order_filter {
public:
    /// \brief Constructor
    fifth_order_filter();

    /// Destructor
    ~fifth_order_filter() {};

    /// \brief Decimate and filter
    /// \param block BBP Block input/output
    /// \param decimate_by Number of points to decimate by
    void decimate(BBP_Block* block, size_t decimate_by = 2);

    /// \brief Decimate and filter
    /// \param data Input data array
    /// \param length length of the input array
    /// \param decimate_by Number of points to decimate
    /// \return number of points afer decimation
    size_t decimate(RADIO_DATA_TYPE* data, size_t length, size_t decimate_by = 2);

private:
    RADIO_DATA_TYPE r_a; ///< Internal real a
    RADIO_DATA_TYPE r_b; ///< Internal real b
    RADIO_DATA_TYPE r_c; ///< Internal real c
    RADIO_DATA_TYPE r_d; ///< Internal real d
    RADIO_DATA_TYPE r_e; ///< Internal real e
    RADIO_DATA_TYPE r_f; ///< Internal real f
    RADIO_DATA_TYPE i_a; ///< Internal imag a
    RADIO_DATA_TYPE i_b; ///< Internal imag b
    RADIO_DATA_TYPE i_c; ///< Internal imag c
    RADIO_DATA_TYPE i_d; ///< Internal imag d
    RADIO_DATA_TYPE i_e; ///< Internal imag e
    RADIO_DATA_TYPE i_f; ///< Internal imag f

    RADIO_DATA_TYPE a; ///< Internal a
    RADIO_DATA_TYPE b; ///< Internal b
    RADIO_DATA_TYPE c; ///< Internal c
    RADIO_DATA_TYPE d; ///< Internal d
};


#endif //RADIONODE_FIFTH_ORDER_FILTER_H
