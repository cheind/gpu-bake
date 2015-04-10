// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef BAKE_CONVERT_SURFACE
#define BAKE_CONVERT_SURFACE

#include <bake/geometry.h>
#include <osg/Node>

namespace bake {
    
    /** Conversion options. */
    enum ConvertOptions {
        ConvertVertexNormals = 1,
        ConvertVertexColors = 2,
        ConvertVertexUVs = 4,
        
        ConvertAll = 0xFFFFFFFF
    };
    
    /** Convert OSG node to internal surface structure. */
    bool convertSurface(const osg::Node *node, Surface &s, unsigned int opts);
    
}

#endif
