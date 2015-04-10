// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

/** This macro alongside an '#include' statement can be used to embed code
    into the application.
 
        char *kernel_source = #include "File.cl";
 
    where the File.cl content is wrapped with the BAKE_STRINGIFY macro.
 */
#define BAKE_STRINGIFY0(x) #x
#define BAKE_STRINGIFY(x) BAKE_STRINGIFY0(x)

#include <string>

namespace bake {
    
    /** Read file to string. */
    std::string readFile(const std::string &path);
    
}
