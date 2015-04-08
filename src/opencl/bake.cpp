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
#include <string>

#include <bake/stringify.h>
std::string clSource =
#include <bake/opencl/bake.cl>

namespace bake {
    namespace opencl {
        
        /** OpenCL context. */
        struct OCL {
            cl::Context ctx;
            cl::Device d;
            cl::Platform p;
            cl::CommandQueue q;
            cl::Program prg;
            cl::Kernel kBakeTexture;
        };
        
        /** Initialize OpenCL relevant structures. */
        bool initOpenCL(OCL &c, int deviceId) {
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
            
            // Build program
            cl::Program::Sources source(1, std::make_pair(clSource.c_str(), clSource.size()));
            c.prg = cl::Program(c.ctx, source, &err);
            err = c.prg.build(devs);
            if (err != CL_SUCCESS) {
                BAKE_LOG("Failed to build OpenCL program: %s", c.prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs.front()).c_str());
                return false;
            }
            
            c.kBakeTexture = cl::Kernel(c.prg, "bakeTextureMap", &err);
            if (err != CL_SUCCESS) {
                BAKE_LOG("Failed to locate kernel");
            }
            
            return true;
            
        }
        
        bool bakeTextureMap(const Surface &from, const Surface &to) {
            OCL ocl;
            initOpenCL(ocl, 1);
            
            // Prepare necessary buffers on GPU
            
            cl::Buffer bVertexPositions(ocl.ctx,
                                        CL_MEM_READ_ONLY,
                                        to.vertexPositions.array().size() * sizeof(float),
                                        const_cast<float*>(to.vertexPositions.data()));
            
            cl::Buffer bVertexUVs(ocl.ctx,
                                  CL_MEM_READ_ONLY,
                                  to.vertexUVs.array().size() * sizeof(float),
                                  const_cast<float*>(to.vertexUVs.data()));
            
            cl::Buffer bVertexNormals(ocl.ctx,
                                      CL_MEM_READ_ONLY,
                                      to.vertexNormals.array().size() * sizeof(float));
            
            // Allocate texture
        
            cl::Image2D bTexture(ocl.ctx,
                                 CL_MEM_WRITE_ONLY,
                                 cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                                 1024,
                                 1024);
            
            ocl.kBakeTexture.setArg(0, bVertexPositions);
            ocl.kBakeTexture.setArg(1, bVertexNormals);
            ocl.kBakeTexture.setArg(2, bVertexUVs);
            ocl.kBakeTexture.setArg(3, bTexture);
            ocl.kBakeTexture.setArg(4, 1024);
            ocl.kBakeTexture.setArg(5, (int)to.vertexPositions.cols()/3);
            
            int nTrianglesDivisableBy2 = to.vertexPositions.cols()/3 + (to.vertexPositions.cols()/3) % 2;
            
            ocl.q.enqueueNDRangeKernel(ocl.kBakeTexture, cl::NullRange, cl::NDRange(nTrianglesDivisableBy2), cl::NullRange);
            ocl.q.finish();
            
            
            
            
            //http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
            
            
            return false;
        }
        
    }
}