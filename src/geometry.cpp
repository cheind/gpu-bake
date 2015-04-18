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
    
    int toIndex(const Eigen::Vector3i &idx, const Eigen::Vector3i &res) {
        return idx.x() + idx.y() * res.x() + idx.z() * res.x() * res.y();
    }
    
    bool buildSurfaceVolume(const Surface &s, const Eigen::Vector3i &voxelsPerDimension, SurfaceVolume &v)
    {
        v.bounds = computeBoundingBox(s.vertexPositions);
        v.voxelsPerDimension = voxelsPerDimension;
        v.voxelSizes = v.bounds.diagonal().array() / voxelsPerDimension.cast<float>().array();
        v.toVoxel = buildWorldToVoxel(v.bounds.min(), v.voxelSizes);
        
        // Loop over triangles and build a sparse map of voxel -> triangles
        
        std::unordered_map<int, std::set<int> > voxelsToTri;
        auto &points = s.vertexPositions.topRows(3);
        
        const int ntri = static_cast<int>(s.vertexPositions.cols() / 3);
        for (int tri = 0; tri < ntri; ++tri) {
            int a = toIndex(toVoxel(v.toVoxel, points.col(tri * 3 + 0)), v.voxelsPerDimension);
            int b = toIndex(toVoxel(v.toVoxel, points.col(tri * 3 + 1)), v.voxelsPerDimension);
            int c = toIndex(toVoxel(v.toVoxel, points.col(tri * 3 + 2)), v.voxelsPerDimension);
            
            voxelsToTri[a].insert(tri);
            voxelsToTri[b].insert(tri);
            voxelsToTri[c].insert(tri);
        }
        
        // Loop over all cells and update output structures.
        v.cells.clear();
        v.triangleIndices.clear();
        
        int next = 0;
        for (int idx = 0; idx < v.voxelsPerDimension.array().size(); ++idx) {
            // Start index in triIndices
            v.cells.push_back(next);
            
            // When cell has triangles add them to list
            auto iter = voxelsToTri.find(idx);
            if (iter != voxelsToTri.end()) {
                v.triangleIndices.insert(v.triangleIndices.end(), iter->second.begin(), iter->second.end());
                next += static_cast<int>(iter->second.size());
            }
            
            // Terminal
            v.triangleIndices.push_back(-1);
            ++next;
        }
        
        return true;
    }




    
    

}