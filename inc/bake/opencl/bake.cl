
// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#pragma OPENCL EXTENSION cl_intel_printf : enable


float cross2(float2 a, float2 b) {
    return a.x*b.y - a.y*b.x;
}

__kernel void bakeTextureMap(
    __global float3* targetVertexPositions,
    __global float3* targetVertexNormals,
    __global float2* targetVertexUVs,
    __global float3* srcVertexPositions,
    __global float3* srcVertexNormals,
    __global float4* srcVertexColors,
    __global int* srcVoxels,
    __global int* srcTrianglesInVoxels,
    float8 srcVoxelBounds,
    float4 srcVoxelSizes,
    int4 srcVoxelPerDimension,
    __write_only image2d_t texture,
    int imageSize,
    float stepOut,
    int nTargetTriangles
)
{
    int triId = get_global_id(0);
    if (triId >= nTargetTriangles) {
        return;
    }
    
    float2 uvA = targetVertexUVs[triId*3+0] * imageSize;
    float2 uvB = targetVertexUVs[triId*3+1] * imageSize;
    float2 uvC = targetVertexUVs[triId*3+2] * imageSize;
    
    float3 xA = targetVertexPositions[triId*3+0];
    float3 xB = targetVertexPositions[triId*3+1];
    float3 xC = targetVertexPositions[triId*3+2];
    
    float3 nA = targetVertexNormals[triId*3+0];
    float3 nB = targetVertexNormals[triId*3+1];
    float3 nC = targetVertexNormals[triId*3+2];
    
    float4 bounds[2];
    bounds[0] = srcVoxelBounds.lo;
    bounds[1] = srcVoxelBounds.hi;
    
    
    // Rasterize triangle in UV space.
    // We currently use a bary-centric approach guided by the UV bounding box of
    // the triangl/Users/cheind/volplay/inc/volplay/rendering/image.he. This wastes
    // performance as it samples the entire bounding box space, but has the advantage
    // that barycentrics are required anyway.
    
    float2 uvMin = max(min(uvA, min(uvB, uvC)), 0.f);
    float2 uvMax = min(max(uvA, max(uvB, uvC)), imageSize - 1.f);

    float inv2A = 1.f / cross2(uvB - uvA, uvC - uvA);

    for (float x = uvMin.x; x <= uvMax.x; x += 0.2f) {
        for (float y = uvMin.y; y <= uvMax.y; y += 0.2f) {
            
            float2 q = (float2)(x,y);
            float u = cross2(uvC - uvB, q - uvB) * inv2A;
            float v = cross2(uvA - uvC, q - uvC) * inv2A;
            float w = 1.f - u - v;
            
            if ((u >= 0) & (v >= 0) & (w >= 0)) {
                // Inside triangle, create ray and march source volume.
                float3 rn = normalize(nA * u + nB * v + nC * w);
                float3 ro = xA * u + xB * v + xC * w;
                ro += rn * stepOut;
                Ray r = createRay(ro, rn * -1.f);
                
                int2 pix = (int2)(round(x), round(y));
                
                float4 color = (float4)(
                    (triId % 255) / 255.f,
                    (triId % 255) / 255.f,
                    (triId % 255) / 255.f,
                    1.f
                );
                
                write_imagef(texture, pix, color);
            }
            
        }
    }
}

