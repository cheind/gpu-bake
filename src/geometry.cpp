// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <bake/geometry.h>
#include <unordered_map>
#include <set>

namespace bake {
    
    Eigen::AlignedBox3f computeBoundingBox(const Surface::VertexPositionMatrix &m) {
        Eigen::AlignedBox3f box;
        
        // For convinience GPU reasons each vector is defined by 4 dims, so
        // we need to extract sub-matrix containing only the first 3 rows.
        auto &vertexPoses = m.topRows(3);
        
        for (Surface::VertexPositionMatrix::Index i = 0; i < vertexPoses.cols(); ++i) {
            box.extend(vertexPoses.col(i));
        }
        return box;
    }
    
    /** Build a transformation that maps from world coordinates to voxel grid coordinates. */
    Eigen::Affine3f buildWorldToVoxel(const Eigen::Vector3f &origin, const Eigen::Vector3f &voxelSizes)
    {
        Eigen::Affine3f a;
        a.setIdentity();
        a.scale(voxelSizes.cwiseInverse()).translate(-origin);
        return a;
    }
    
    Eigen::Vector3i toVoxel(const Eigen::Affine3f &wl, const Eigen::Vector3f &x)
    {
        Eigen::Vector3f l = wl * x;
        return Eigen::Vector3i((int)floor(l.x()),
                               (int)floor(l.y()),
                               (int)floor(l.z()));
    }
    
    int toIndex(const Eigen::Vector3i &idx, int voxelResolution) {
        return idx.x() + idx.y() * voxelResolution + idx.z() * voxelResolution * voxelResolution;
    }
    
    bool buildSurfaceVolume(const Surface &s, int nVoxelsPerDimension, std::vector<int> &cells, std::vector<int> &triIndices)
    {
        // Loop over triangles and build a sparse map of voxel -> triangles
        std::unordered_map<int, std::set<int> > voxelsToTri;
        
        Eigen::Vector3f voxelSizes = s.bounds.diagonal() / nVoxelsPerDimension;
        Eigen::Affine3f t = buildWorldToVoxel(s.bounds.min(), voxelSizes);
        
        auto &points = s.vertexPositions.topRows(3);
        
        const int ntri = static_cast<int>(s.vertexPositions.cols() / 3);
        for (int tri = 0; tri < ntri; ++tri) {
            int a = toIndex(toVoxel(t, points.col(tri * 3 + 0)), nVoxelsPerDimension);
            int b = toIndex(toVoxel(t, points.col(tri * 3 + 1)), nVoxelsPerDimension);
            int c = toIndex(toVoxel(t, points.col(tri * 3 + 2)), nVoxelsPerDimension);
            
            voxelsToTri[a].insert(tri);
            voxelsToTri[b].insert(tri);
            voxelsToTri[c].insert(tri);
        }
        
        // Loop over all cells and update output structures.
        cells.clear();
        triIndices.clear();
        
        int next = 0;
        for (int idx = 0; idx < nVoxelsPerDimension*nVoxelsPerDimension*nVoxelsPerDimension; ++idx) {
            // Start index in triIndices
            cells.push_back(next);
            
            // When cell has triangles add them to list
            auto iter = voxelsToTri.find(idx);
            if (iter != voxelsToTri.end()) {
                triIndices.insert(triIndices.end(), iter->second.begin(), iter->second.end());
                next += static_cast<int>(iter->second.size());
            }
            
            // Terminal
            triIndices.push_back(-1);
            ++next;
        }
        
        return true;
    }




    
    

}