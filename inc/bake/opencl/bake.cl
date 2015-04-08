
BAKE_STRINGIFY(

// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#pragma OPENCL EXTENSION cl_intel_printf :enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

float cross2(float2 a, float2 b) {
    return a.x*b.y - a.y*b.x;
}

__kernel void bakeTextureMap(
    __global float4* toVertexPositions,
    __global float4* toVertexNormals,
    __global float2* toVertexUVs,
    __write_only image2d_t texture,
    float imageSize,
    int nTriangles
)
{
    int triId = get_global_id(0);
    if (triId >= nTriangles) {
        return;
    }
    
    float2 uvA = toVertexUVs[triId*3+0] * imageSize;
    float2 uvB = toVertexUVs[triId*3+1] * imageSize;
    float2 uvC = toVertexUVs[triId*3+2] * imageSize;
    
    // Rasterize triangle in UV space.
    // We currently use a bary-centric approach guided by the UV bounding box of
    // the triangle. This wastes performance as it samples the entire bounding box
    // space, but has the advantage that barycentrics are required anyway.
    
    float2 uvMax = max(uvA, max(uvB, uvC));
    float2 uvMin = min(uvA, min(uvB, uvC));
    
    float2 uvBA = uvB - uvA;
    float2 uvCA = uvC - uvA;
    float invCrossBACA = 1.f / cross2(uvBA, uvCA);

    for (float x = uvMin.x; x <= uvMax.x; ++x) {
        for (float y = uvMin.y; y <= uvMax.y; ++y) {
            float2 e = (float2)(x - uvA.x, y - uvB.y);
            float s = cross2(e, uvCA) * invCrossBACA;
            float t = cross2(uvBA, e) * invCrossBACA;
            float u = 1.f - s - t;
            printf("%f %f\n", s, t);
            if ((s >= 0) & (t >= 0) & (u >= 0)) {
                
                write_imagef(texture, (int2)(x, y), (float4)(1.f));
            }
            
        }
    }
}

);

