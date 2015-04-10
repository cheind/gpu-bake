
// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

float cross2(float2 a, float2 b) {
    return a.x*b.y - a.y*b.x;
}

__kernel void bakeTextureMap(
    __global float4* toVertexPositions,
    __global float4* toVertexNormals,
    __global float2* toVertexUVs,
    __write_only image2d_t texture,
    int imageSize,
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
    // the triangl/Users/cheind/volplay/inc/volplay/rendering/image.he. This wastes
    // performance as it samples the entire bounding box space, but has the advantage
    // that barycentrics are required anyway.
    
    float2 uvMin = max(min(uvA, min(uvB, uvC)), 0.f);
    float2 uvMax = min(max(uvA, max(uvB, uvC)), imageSize - 1.f);

    float invA = 1.f / fabs(cross2(uvC - uvB, uvB - uvA));

    for (float x = uvMin.x; x <= uvMax.x; x += 0.2f) {
        for (float y = uvMin.y; y <= uvMax.y; y += 0.2f) {
            
            float2 q = (float2)(x,y);
            float u = cross2(uvC - uvB, q - uvB) * invA;
            float v = cross2(uvA - uvC, q - uvC) * invA;
            float w = 1.f - u - v;
            
            if ((u >= 0) & (v >= 0) & (w >= 0)) {
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

