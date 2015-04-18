
// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

// https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
typedef struct {
    float3 o;
    float3 d;
    float3 invd;
    int3 sign;
} Ray;

Ray createRay(float3 origin, float3 dir) {
    Ray r;
    r.o = origin;
    r.d = dir;
    r.invd = (float3)(1.f) / dir;
    r.sign = select((int3)(0),(int3)(1),r.invd < 0.f);
    return r;
}

/** Intersect ray / box. Returns entry, exit values for parametric t.
    When ret.x > ret.y no interesection occurred. */
float2 intersectRayBox(Ray ray, float3 aabb[2]) {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    
    tmin = (aabb[ray.sign.x].x - ray.o.x) * ray.invd.x;
    tmax = (aabb[1-ray.sign.x].x - ray.o.x) * ray.invd.x;
    tymin = (aabb[ray.sign.y].y - ray.o.y) * ray.invd.y;
    tymax = (aabb[1-ray.sign.y].y - ray.o.y) * ray.invd.y;
    tzmin = (aabb[ray.sign.z].z - ray.o.z) * ray.invd.z;
    tzmax = (aabb[1-ray.sign.z].z - ray.o.z) * ray.invd.z;
    tmin = max(max(tmin, tymin), tzmin);
    tmax = min(min(tmax, tymax), tzmax);
    
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
