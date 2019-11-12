//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_BBP_BLOCK_H
#define RADIONODE_BBP_BLOCK_H

#include <common/Common_Deffinitions.h>
#include <string.h>

extern const Complex NULLPOINT;

struct BBP_Block
{
    Complex_Array points;   /// point array real/imag points
    size_t n_points;        /// Actual number of points in the array

    BBP_Block()
    : points(NULLPOINT, BLOCK_READ_SIZE)
    , n_points(0)
    {
        reset();
    }

    BBP_Block(size_t size)
    : points(NULLPOINT, size)
    , n_points(0)
    {
        reset();
    }

    size_t full_size(void) const
    {
        return sizeof(BBP_Block);
    }

    size_t number_of_points() const
    {
        return n_points;
    }

    size_t number_of_points_byte_size() const
    {
        return n_points*sizeof(Complex);
    }

    size_t empty_space() const
    {
        return BLOCK_READ_SIZE - n_points;
    }

    void reset()
    {
        n_points = 0;
        memset(&points[0], 0, points.size()*sizeof(Complex));
    }

    void hard_copy(const BBP_Block &block)
    {
        memcpy(&points[0], &block.points[0], sizeof(Complex_Array));
        n_points = block.n_points;
    }
};


#endif //RADIONODE_BBP_BLOCK_H
