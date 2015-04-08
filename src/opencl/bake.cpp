// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <bake/opencl/bake.h>
#include <bake/opencl/cl.hpp>
#include <bake/log.h>
#include <vector>

namespace bake {
    namespace opencl {
        
        struct Context {
            cl::Context ctx;
            cl::Device d;
            cl::Platform p;
            cl::CommandQueue q;
        };
        
        bool createContext(Context &c, int deviceId) {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            
            if (platforms.empty()) {
                BAKE_LOG("No OpenCL compatible platforms found.");
                return false;
            }
            
            bool deviceFound = false;
            int id = 0;
            for (auto piter = platforms.begin(); piter != platforms.end(); ++piter) {
                std::vector<cl::Device> devices;
                piter->getDevices(CL_DEVICE_TYPE_ALL, &devices);
                for (auto citer = devices.begin(); citer != devices.end(); ++citer) {
                    BAKE_LOG("Found device #%d: %s", id, citer->getInfo<CL_DEVICE_NAME>().c_str());
                    if (id == deviceId) {
                        c.p = *piter;
                        c.d = *citer;
                        deviceFound = true;
                    }
                    ++id;
                }
            }
            
            if (!deviceFound) {
                BAKE_LOG("Requested device not found.");
            }
            
            BAKE_LOG("Using device %s.", c.d.getInfo<CL_DEVICE_NAME>().c_str());
            
            cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(c.p)(), 0};
            std::vector<cl::Device> devs;
            devs.push_back(c.d);
            
            cl_int err;
            c.ctx = cl::Context(devs, properties, 0, 0, &err);
            if (err != CL_SUCCESS) {
                BAKE_LOG("Failed to create OpenCL context.");
            }
            
            c.q = cl::CommandQueue(c.ctx, c.d, 0, &err);
            if (err != CL_SUCCESS) {
                BAKE_LOG("Failed to create OpenCL queue.");
            }
            
            return true;
            
        }
        
        bool bakeTextureMap(const Surface &from, const Surface &to) {
            Context c;
            createContext(c, 0);
            
            
            return false;
        }
        
    }
}