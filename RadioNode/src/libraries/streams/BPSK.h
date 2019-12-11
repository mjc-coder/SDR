/// @file SDR/RadioNode/src/libraries/streams/BPSK.h


#ifndef RADIONODE_BPSK_H
#define RADIONODE_BPSK_H

#include <string>
#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <common/confidence_counter.h>

using namespace std;

#ifndef SIGN_FUNC
/// Defines the SIGN function.  val > 0 = 1 or val < 0 = -1 or val = 0 then 0
#define SIGN_FUNC(val) (( val == 0.0 ) ? 0.0 : ( (val < 0.0) ? -1.0 : 1.0))
#endif

/// \brief Binary Phase Shift Keying Modulation and Demodulation
class BPSK
{
public:
    static const RADIO_DATA_TYPE LUT_REAL[];    ///< Look up table Real
    static const RADIO_DATA_TYPE LUT_180[];     ///< Look up Table 180 Degree Phase Shift
    static const RADIO_DATA_TYPE RRC_COEFF[];   ///< Root Raise Coefficients
public:
    /// \brief Enumeration for each of the possible Phase Ambiguities that can occur
    enum PhaseAmbiguity
    {
        Ambiguity_0   = 0,  ///< Phase Ambiguity of 0 degrees
        Ambiguity_180 = 1   ///< Phase Ambiguity of 180 degrees
    };

public:
    /// \brief Constructor
    /// \param UniqueWord Unique word that is prepended on each packet.
    /// \param UniqueWordLength Length in bits of the Unique Word
    BPSK(uint64_t UniqueWord = 0x4E,
         size_t UniqueWordLength = 8);

    /// \brief Destructor
    ~BPSK();

    /// \brief Modulate the input data stream of binary bits (1's and 0's)
    /// \param data_in input stream of bits (1's and 0's)
    /// \param length Length of the input data stream
    /// \param modulated_real modulated real stream of data
    /// \param length_modulated length of the modulated stream
    void modulate(uint8_t* data_in, size_t length, RADIO_DATA_TYPE* modulated_real, size_t length_modulated);

    /// \brief Demodulate the complex stream of data.
    /// \param input input stream of complex data
    /// \param output output stream of complex data
    void demodulate(BBP_Block* input, BBP_Block* output);

    /// \brief Demodulate the complex input stream of data
    /// \param input Input stream of complex data
    /// \param output output stream of ones and zeros
    /// \param number_of_points Total number of points in the input streams
    /// \param downsample decimation factor for the downsampling
    /// \param real_plot_data Real data for debug purposes only
    /// \param imag_plot_data Imag data for debug purposes only
    /// \return total number of demodulated bits
    size_t demodulate(Complex_Array& input, uint8_t* output, size_t number_of_points, size_t downsample, RADIO_DATA_TYPE* real_plot_data, RADIO_DATA_TYPE* imag_plot_data);


    /// \brief Encodes the input stream of bits from 0's and 1's to -1's and 1's respectively.
    /// \details Modulation function
    /// \param data_in Input binary stream. Encoded data is stored internally.
    /// \param length Length of the input stream.
    void encode_LUT(uint8_t* data_in, size_t length);

    /// \brief Encode the binary data with the pulse shape
    /// \details Modulate function
    /// \param output_data  Modulated output stream of data
    /// \param output_length length of the output buffer
    void encode_pulseShape(RADIO_DATA_TYPE* output_data, size_t output_length);


    /// \brief Decode the pulse shape.
    /// \details Demodulate function
    /// \param input Input stream of complex data
    /// \param real Output stream of Real data points
    /// \param imag Output stream of imag data points
    /// \return Total number of points in the Real/Imag stream.
    size_t decode_pulseShape(BBP_Block* input, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag);

    /// \brief Decode the pulse shape
    /// \details Demodulate function
    /// \param input Input stream of complex data
    /// \param number_of_points Number of points in the Input stream.
    /// \param real Output stream of real data
    /// \param imag Output stream of Imag data
    /// \return Number of points in the Output streams
    size_t decode_pulseShape(Complex_Array& input, size_t number_of_points, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag);

    /// \brief Symbol Timing Synchronization routine
    /// \details Demodulate function
    /// \param real Real input stream of data
    /// \param imag Imag input stream of data
    /// \param num_of_points total number of points
    void symbol_timing_synchronization(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);

    /// \brief Phase Shift Correction routine
    /// \details Demodulate function
    /// \param real Input/Output stream data points
    /// \param imag Input/Output stream data points
    /// \param num_of_points Total number of points
    /// \return number of points after phase shift correction
    size_t phase_shift_correction(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);

    /// \brief Phase ambiguity correction
    /// \details Demodulate function
    /// \param real Input/Output stream of data points
    /// \param imag Input/Output stream of data points
    /// \param num_of_points Number of points
    void phase_ambiguity_correction(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);

    /// \brief Complex multiplication routine A * B = C
    /// \param ra Real A
    /// \param ia Imag A
    /// \param rb Real B
    /// \param ib Imag B
    /// \param rc Real C
    /// \param ic Imag C
    void complex_multiply(RADIO_DATA_TYPE ra, RADIO_DATA_TYPE ia, RADIO_DATA_TYPE rb, RADIO_DATA_TYPE ib, RADIO_DATA_TYPE& rc, RADIO_DATA_TYPE& ic);


    /// \brief Internal Symbol Timing Synchronization variables
    struct SymbolTimingSynchronizationVars
    {
        RADIO_DATA_TYPE mu; ///< Mu Value
        RADIO_DATA_TYPE I; ///< I Value
        RADIO_DATA_TYPE Q; ///< Q Value
        RADIO_DATA_TYPE muDelay; ///< Mu Delay Value

        /// \brief Constructor / Struct Initializer
        SymbolTimingSynchronizationVars()
        : mu(0)
        , I(0)
        , Q(0)
        , muDelay(0)
        {
        }
    } SymbolTimingSynchronizationVariables; ///< Single instance for internal use

    /// \brief Internal Interpolation Data
    struct InterpData
    {
        RADIO_DATA_TYPE delay1; ///< Delay 1
        RADIO_DATA_TYPE delay2; ///< Delay 2
        RADIO_DATA_TYPE delay3; ///< Delay 3
        RADIO_DATA_TYPE delay4; ///< Delay 4
        RADIO_DATA_TYPE delay5; ///< Delay 5

        /// Constructor / Initializer
        InterpData()
        : delay1(0)
        , delay2(0)
        , delay3(0)
        , delay4(0)
        , delay5(0)
        {
        }
    };
    InterpData Interp_I_Channel;    ///< I Channel Interpolation
    InterpData Interp_Q_Channel;    ///< Q Channel Interpolation

    /// \brief Interpolation Filter Routine
    /// \param data Structure of internal data
    /// \param in Input data value
    /// \param mu Mu Value
    /// \return Output data value after Interpolation filter
    RADIO_DATA_TYPE InterpolationFilter(InterpData& data, RADIO_DATA_TYPE in, RADIO_DATA_TYPE mu);

    /// \brief Internal Gardner Detector data
    struct GardnerDetectorData
    {
        RADIO_DATA_TYPE delay1; ///< Delay 1
        RADIO_DATA_TYPE delay2; ///< Delay 2
        RADIO_DATA_TYPE delay3; ///< Delay 3
        RADIO_DATA_TYPE delay4; ///< Delay 4

        /// Constructor / Initializer
        GardnerDetectorData()
        : delay1(1)
        , delay2(1)
        , delay3(1)
        , delay4(1)
        {
        }
    } GardnerDetectorVariables; ///< Internal Instance of Gardner Detector Variables

    /// \brief Gardner detector Routine
    /// \param i Input I value
    /// \param Q Input Q Value
    /// \param strobe Stobe Value
    /// \return Calculated value
    RADIO_DATA_TYPE GardnerDetector(RADIO_DATA_TYPE i, RADIO_DATA_TYPE Q, RADIO_DATA_TYPE strobe);

    /// \brief Internal Loop Filter variables
    struct LoopFilterVars
    {
        RADIO_DATA_TYPE delay_x;    ///< X delay value
        RADIO_DATA_TYPE delay_y;    ///< Y delay value

        /// \brief Loop Filter Constructor / Initializer
        LoopFilterVars()
        : delay_x(0)
        , delay_y(0)
        {
        }
    } LoopFilterVariables; ///< Internal Instance of Loop Filter Variables

    /// \brief Loop filter routine
    /// \param val Input Value
    /// \return Output value
    RADIO_DATA_TYPE LoopFilter(RADIO_DATA_TYPE val);

    /// \brief NCO Control Variables
    struct NcoControlVars
    {
        RADIO_DATA_TYPE delay;  ///< Delay Variable

        /// Constructor / Initializer
        NcoControlVars()
        : delay(0)
        {
        }
    } NcoControlVariables; ///< Internal Instance of NCO Control Variables

    /// \brief NCO Control routine
    /// \param input Input value
    /// \return Output value
    RADIO_DATA_TYPE NcoControl(RADIO_DATA_TYPE input);

    /// \brief Strobe Enabled Register Routine
    /// \param input Input value
    /// \param strobe Strobe input value
    /// \param reg Register value that is updated when strobe is high
    /// \return Output data value
    RADIO_DATA_TYPE StrobeEnabledRegister(RADIO_DATA_TYPE input, RADIO_DATA_TYPE strobe, RADIO_DATA_TYPE& reg);
    RADIO_DATA_TYPE Register_mu;    ///< Register Mu value
    RADIO_DATA_TYPE Register_I;     ///< Register I value
    RADIO_DATA_TYPE Register_Q;     ///< Register Q value


    /// \brief Internal Phase Correction Variables
    struct PhaseCorrectionVars
    {
        RADIO_DATA_TYPE dds_real;   ///< DDS Real value
        RADIO_DATA_TYPE dds_imag;   ///< DDS Imag value
        RADIO_DATA_TYPE dds_delay;  ///< DDS delay value

        /// \brief Constructor / Initializer
        PhaseCorrectionVars()
        : dds_real(1)
        , dds_imag(0)
        , dds_delay(0)
        {
        }
    } PhaseCorrectionVariables; ///< Internal Instance of Phase Correction Variables

    /// \brief Phase loop filter variables
    struct PhaseLoopFilterVars
    {
        RADIO_DATA_TYPE delay_x;    ///< Delay X
        RADIO_DATA_TYPE delay_y;    ///< Delay Y

        /// \brief Constructor / Initializer
        PhaseLoopFilterVars()
        : delay_x(0)
        , delay_y(0)
        {
        }
    } PhaseLoopFilterVariables; ///< Internal Instance of Phase Loop Filter Variables

    /// \brief Phase loop filter routine
    /// \param val Input value
    /// \return Output value
    RADIO_DATA_TYPE PhaseLoopFilter(RADIO_DATA_TYPE val);

    /// \brief Phase Shift Correction Output Loopkup
    /// \param real Real Input data
    /// \param imag Imag Input Data
    /// \return Output data
    RADIO_DATA_TYPE PhaseShiftCorrection_OutputLookup(RADIO_DATA_TYPE real, RADIO_DATA_TYPE imag);

    /// \brief Simple routine to modulus an Angle to 360 degrees
    /// \param x Input X value
    /// \return Wrapped value (0 - 2Pi)
    inline RADIO_DATA_TYPE constrainAngle(RADIO_DATA_TYPE x)
    {
        x = fmod(x,2.0*PI);
        if (x < 0.0)
            x += 2.0*PI;
        return x;
    }

    /// Other variables
    RADIO_DATA_TYPE* m_RealData; ///< Internal Real data pointer
    size_t m_length; ///< Length of the internal buffer
    RADIO_DATA_TYPE normalization_max; ///< Normalization Max value
    confidence_counter m_phase_ambiguity_cc; ///< Phase ambiguity confidence counter
    RADIO_DATA_TYPE m_differential_decode_delay;    ///< Differential Decode Delay
    size_t UW_SIZE; ///< Size of the Unique Word
    RADIO_DATA_TYPE uw_0[64];   ///< Unique word with 0 degree phase shift
    RADIO_DATA_TYPE uw_180[64]; ///< Unique word with 180 degree phase shift
};


#endif //RADIONODE_BPSK_H
