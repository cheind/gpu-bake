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
        Eigen::AlignedBox3f bounds;
    };
    
    /** Compute an axis aligned bounding box for the given points. */
    Eigen::AlignedBox3f computeBoundingBox(const Surface::VertexPositionMatrix &m);
    
    /** Build a transformation that maps from world coordinates to voxel grid coordinates. */
    Eigen::Affine3f buildWorldToVoxel(const Eigen::Vector3f &origin, const Eigen::Vector3f &voxelSizes);

    /** Transform world point to voxel it falls into. */
    Eigen::Vector3i toVoxel(const Eigen::Affine3f &wl, const Eigen::Vector3f &x);
    
    /** Map voxel index to flat array index. */
    int toIndex(const Eigen::Vector3i &idx, int res);
    
    /** Builds a uniform grid where each voxel maps to all triangle indices intersecting that voxel. */
    bool buildSurfaceVolume(const Surface &s, int nVoxelsPerDimension, std::vector<int> &cells, std::vector<int> &triIndices);
    
}

#endif
