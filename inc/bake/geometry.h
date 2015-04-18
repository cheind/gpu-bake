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
#include <vector>

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
    
    /**
        Uniform grid volume over a triangle mesh.
     
        Stores a single index per voxel that represents the first trianlge index in that voxel.
        All triangle indices are considered to be part of the cell until a terminator index is found (-1).
    */
    struct SurfaceVolume {
        Eigen::AlignedBox3f bounds;
        Eigen::Affine3f toVoxel;
        Eigen::Vector3i voxelsPerDimension;
        Eigen::Vector3f voxelSizes;
        std::vector<int> cells;
        std::vector<int> triangleIndices;
    };
    
    /** Compute an axis aligned bounding box for the given points. */
    Eigen::AlignedBox3f computeBoundingBox(const Surface::VertexPositionMatrix &m);
    
    /** Builds a uniform grid where each voxel maps to all triangle indices intersecting that voxel. */
    bool buildSurfaceVolume(const Surface &s, const Eigen::Vector3i &voxelsPerDimension, SurfaceVolume &v);
    
}

#endif
