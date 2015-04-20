
// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#pragma OPENCL EXTENSION cl_intel_printf : enable

// https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
typedef struct {
    float3 o;
    float3 d;
    float3 invd;
} Ray;

Ray createRay(float3 origin, float3 dir) {
    Ray r;
    r.o = origin;
    r.d = dir;
    r.invd = (float3)(1.f) / dir;
    return r;
}

/** Intersect ray / box. Returns entry, exit values for parametric t.
    When ret.x > ret.y no interesection occurred. 
 https://code.google.com/p/rtrt-on-gpu/source/browse/trunk/Source/GLSL+Tutorial/Implicit+Surfaces/Fragment.glsl?r=305
 */
float2 intersectRayBox(Ray ray, float3 aabb[2]) {
    
    float3 omin = (aabb[0] - ray.o) / ray.d;
    float3 omax = (aabb[1] - ray.o) / ray.d;
    float3 mmax = max(omax, omin);
    float3 mmin = min(omax, omin);
    
    float tmax = min(mmax.x, min(mmax.y, mmax.z));
    float tmin = max(max(mmin.x, 0.f), max(mmin.y, mmin.z));
    
    return (float2)(tmin, tmax);
}

// https://courses.cs.washington.edu/courses/csep557/10au/lectures/triangle_intersection.pdf
float3 intersectRayTriangle(Ray r, float3 a, float3 b, float3 c) {
    
    // Two steps: Intersect with triangle plane, then use bary-centric test
    float3 un = cross(b - a, c - a);
    float3 n = normalize(un);
    float w = dot(n, a);
    float d = dot(n, r.d);
    
    if (d == 0.f)
        return (float3)(-1.f);
        
        
    float t = (w - dot(n, r.o)) / d;
    if (t < 0.f)
        return (float3)(-1.f);
    
    float3 q = r.o + r.d * t;
    float invAreaABC = 1.f / dot(un, n);
    float alpha = dot(cross(c - b, q - b), n) * invAreaABC;
    float beta = dot(cross(a - c, q - c), n) * invAreaABC;
    
    if ((alpha >= 0) & (beta >= 0) & (alpha + beta <= 1.f))
        return (float3)(t, alpha, beta);
    else
        return (float3)(-1.f);
}

void findTriangleInVoxel(
    Ray r,
    int3 voxelIdx,
    int3 voxelResolution,
    __global float3 *positions,
    __global int* voxels,
    __global int* trisInVoxels,
    __private int *triIdx,
    __private float3 *triHit)
{
    int id = voxelIdx.x + voxelIdx.y * voxelResolution.x + voxelIdx.z * voxelResolution.x * voxelResolution.y;
    
    int triListIndex = voxels[id];
    int triId = trisInVoxels[triListIndex];
    
    float3 bestHit = float3(FLT_MAX, FLT_MAX, FLT_MAX);
    int bestTri = -1;
    
    while (triId != -1) {
        
        float3 hit = intersectRayTriangle(r, positions[triId * 3 + 0], positions[triId * 3 + 1], positions[triId * 3 + 2]);
        bool closest = (hit.x >= 0 & hit.x < bestHit.x);
        bestHit = closest ? hit : bestHit;
        bestTri = closest ? triId : bestTri;

        triListIndex += 1;
        triId = trisInVoxels[triListIndex];
    }
    
    *triIdx = bestTri;
    *triHit = bestHit;
}

void ddaTriangleVolume(
    Ray r,
    float3 aabb[2],
    float3 voxelSizes,
    float3 invVoxelSizes,
    int3 voxelResolution,
    __global float3 *positions,
    __global int* voxels,
    __global int* trisInVoxels,
    __private int *triIdx,
    __private float3 *triHit)
{
    *triIdx = -1;
    *triHit = -1.f;
    
    float2 tRange = intersectRayBox(r, aabb);
    if (tRange.x > tRange.y)
        return;

    // https://www-s.ks.uiuc.edu/Research/vmd/projects/ece498/raytracing/RTonGPU.pdf
    // Adjust ray to start inside volume.
    r.o += r.d * tRange.x;
    
    float3 voxel = (r.o - aabb[0]) * invVoxelSizes;
    int3 voxelIdx = convert_int3_rtz(voxel);
    voxelIdx = clamp(voxelIdx, (int3)(0), voxelResolution-1);

    float3 voxelMin = aabb[0] + voxelIdx * voxelSizes;
    float3 voxelMax = aabb[0] + (voxelIdx + (int3)(1)) * voxelSizes;
    float3 maxNeg = (voxelMin - r.o) * r.invd;
    float3 maxPos = (voxelMax - r.o) * r.invd;
    float3 tmax = (r.d < 0.f) ? maxNeg : maxPos;
    tmax = (fabs(r.d) < 1e-5f) ? (float3)(FLT_MAX) : tmax;
    int3 step = (r.d < 0) ? (int3)(-1) : (int3)(1);
    float3 tdelta = fabs(voxelSizes * r.invd);
    
    while ((*triIdx == -1) & all(voxelIdx >= 0) & all(voxelIdx < voxelResolution))
    {
        
        // Find intersected triangle
        findTriangleInVoxel(r, voxelIdx, voxelResolution, positions, voxels, trisInVoxels, triIdx, triHit);
        
        // Instead of ifs:
        //http://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/grid_8cpp_source.html
        
        // Advance to the next cell along the ray using 3D-DDA.
        if (tmax.x < tmax.y)
        {
            if (tmax.x < tmax.z)
            {
                voxelIdx.x += step.x;
                tmax.x += tdelta.x;
            }
            else
            {
                voxelIdx.z += step.z;
                tmax.z += tdelta.z;
            }
        }
        else
        {
            if (tmax.y < tmax.z)
            {
                voxelIdx.y += step.y;
                tmax.y += tdelta.y;
            }
            else
            {
                voxelIdx.z += step.z;
                tmax.z += tdelta.z;
            }
        }
    }
    
}


