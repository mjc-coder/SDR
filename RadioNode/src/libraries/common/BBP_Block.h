/// @file SDR/RadioNode/src/libraries/common/BBP_Block.h

#ifndef RADIONODE_BBP_BLOCK_H
#define RADIONODE_BBP_BLOCK_H

#include <common/Common_Deffinitions.h>
#include <string.h>

extern const Complex NULLPOINT; ///< NULL Complex Data Point

/// BBP Black is a baseband point block of complex data points.
struct BBP_Block
{
    Complex_Array points;   ///< point array real/imag points
    size_t n_points;        ///< Actual number of points in the array

    /// \brief Constructor
    BBP_Block()
    : points(NULLPOINT, BLOCK_READ_SIZE)
    , n_points(0)
    {
        reset();
    }

    /// \brief Constructor with given size
    /// \param size Number of points to initialize the BBP Block with
    BBP_Block(size_t size)
    : points(NULLPOINT, size)
    , n_points(0)
    {
        reset();
    }

    /// \brief Get the total number of points
    /// \return number of points in BBP Block
    size_t number_of_points() const
    {
        return n_points;
    }

    /// \brief Clears the BBP Block
    void reset()
    {
        n_points = 0;
        memset(&points[0], 0, points.size()*sizeof(Complex));
    }

    /// \brief Hard Copy the given block to this one.
    /// \param block Block to copy
    void hard_copy(const BBP_Block &block)
    {
        memcpy(&points[0], &block.points[0], sizeof(Complex_Array));
        n_points = block.n_points;
    }
};


#endif //RADIONODE_BBP_BLOCK_H
