// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef BAKE_SURFACE
#define BAKE_SURFACE

#include <Eigen/Dense>

namespace bake {
    
    /** 
        Defines a triangulated surface.
     
        Triangles are defines by consecutive triples of matrix columns. This might be improved
        towards lesser memory footprint in future work. 
    */
    struct Surface {
        typedef Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> VertexPositionMatrix;
        typedef Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> VertexColorMatrix;
        typedef Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> VertexNormalMatrix;
        typedef Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> VertexUVMatrix;
        
        VertexPositionMatrix vertexPositions;
        VertexColorMatrix vertexColors;
        VertexNormalMatrix vertexNormals;
        VertexUVMatrix vertexUVs;
    };
}

#endif
