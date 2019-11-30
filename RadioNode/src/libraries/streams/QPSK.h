/// @file SDR/RadioNode/src/libraries/streams/QPSK.h


#ifndef RADIONODE_QPSK_H
#define RADIONODE_QPSK_H

#include <string>
#include <common/BBP_Block.h>
#include <common/Common_Deffinitions.h>
#include <common/confidence_counter.h>

using namespace std;

#ifndef SIGN_FUNC
#define SIGN_FUNC(val) (( val == 0.0 ) ? 0.0 : ( (val < 0.0) ? -1.0 : 1.0))
#endif

/// \brief Binary Phase Shift Keying Modulation and Demodulation
class QPSK
{
public:
    static const RADIO_DATA_TYPE LUT_REAL[];    ///< Look up Table for Real Data
    static const RADIO_DATA_TYPE LUT_IMAG[];    ///< Look up Table for Imag DAta
    static const RADIO_DATA_TYPE LUT_REAL___0[]; ///< Look up Table for 0 degree phase shift
    static const RADIO_DATA_TYPE LUT_REAL__90[]; ///< Look up Table for 90 degree phase shift
    static const RADIO_DATA_TYPE LUT_REAL_180[]; ///< Look up Table for 180 degree phase shift
    static const RADIO_DATA_TYPE LUT_REAL_270[]; ///< Look up Table for 270 degree phase shift
    static const RADIO_DATA_TYPE RRC_COEFF[]; ///< Root Raised Cosigned Coefficients
    static const RADIO_DATA_TYPE DEFAULT_UNIQUE_WORD[64];   ///< Default Unique Word
public:
    /// \brief Enumeration for each of the possible Phase Ambiguities that can occur
    enum PhaseAmbiguity
    {
        Ambiguity_0   = 0,  ///< 0 degree phase shift
        Ambiguity_180 = 1   ///< 180 degree phase shift
    };

public:
    /// \brief Constructor
    /// \param UniqueWord Unique Word input
    /// \param UniqueWordLength Length of the Unique Word in bits
    QPSK(const RADIO_DATA_TYPE* UniqueWord = DEFAULT_UNIQUE_WORD, //0xEB926C63,
         size_t UniqueWordLength = 16);

    /// \brief Destructor
    ~QPSK();

    /// \brief Modulate the input data stream of binary bits (1's and 0's)
    /// \param data_in input stream of bits (1's and 0's)
    /// \param length Length of the input data stream
    /// \param modulated_real modulated real stream of data
    /// \param modulated_imag modulated imag stream of data
    /// \param length_modulated length of the modulated stream
    void modulate(uint8_t* data_in, size_t length, RADIO_DATA_TYPE* modulated_real, RADIO_DATA_TYPE* modulated_imag, size_t length_modulated);

    /// \brief Demodulate the complex stream of data.
    /// \param input input stream of complex data
    /// \param outputdemodulated output stream of complex data
    void demodulate(BBP_Block* input, BBP_Block* output);

    /// \brief Encodes the input stream of bits from 0's and 1's to -1's and 1's respectively.
    /// \details Modulation function
    /// \param data_in Input binary stream. Encoded data is stored internally.
    /// \param length Length of the input stream.
    void encode_LUT(uint8_t* data_in, size_t length);

    /// \brief Encode the binary data with the pulse shape
    /// \details Modulate function
    /// \param real_data Modulated output stream of real data
    /// \param imag_data Moudlated output stream of imag data
    /// \param output_length length of the output buffer
    void encode_pulseShape(RADIO_DATA_TYPE* real_data, RADIO_DATA_TYPE* imag_data, size_t output_length);

    /// \brief Decode the pulse shape.
    /// \details Demodulate function
    /// \param input Input stream of complex data
    /// \return Total number of points in the Real/Imag stream.
    size_t decode_pulseShape(BBP_Block* input);

    /// \brief Symbol Timing Synchronization routine
    /// \details Demodulate function
    void symbol_timing_synchronization();

    /// \brief Phase Shift Correction routine
    /// \details Demodulate function
    void phase_shift_correction();

    /// \brief Phase ambiguity correction
    /// \details Demodulate function
    void phase_ambiguity_correction();

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
        RADIO_DATA_TYPE muDelay; /// Mu Delay Value

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
    RADIO_DATA_TYPE* m_RealData;
    RADIO_DATA_TYPE* m_ImagData;
    size_t m_length;
    RADIO_DATA_TYPE normalization_max;
    PhaseAmbiguity m_phase_ambiguity_shift;
    confidence_counter m_phase_ambiguity_cc_000;
    confidence_counter m_phase_ambiguity_cc_090;
    confidence_counter m_phase_ambiguity_cc_180;
    confidence_counter m_phase_ambiguity_cc_270;
    RADIO_DATA_TYPE m_differential_decode_delay;
    size_t UW_SIZE; // Size is the same for all
    RADIO_DATA_TYPE uw_000[64];
    RADIO_DATA_TYPE uw_090[64];
    RADIO_DATA_TYPE uw_180[64];
    RADIO_DATA_TYPE uw_270[64];
};


#endif //RADIONODE_QPSK_H
