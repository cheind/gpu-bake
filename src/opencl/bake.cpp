// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <bake/opencl/bake.h>
#include <bake/opencl/cl.hpp>
#include <bake/stringify.h>
#include <bake/geometry.h>
#include <bake/log.h>
#include <bake/image.h>
#include <bake/config.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#define ASSERT_OPENCL(clerr, msg)           \
if (clerr != CL_SUCCESS) {              \
BAKE_LOG("%s : %d", msg, clerr);    \
return false;                       \
}

namespace cl {
    namespace detail {
        
        /** Argument type for local c-arrays */
        struct carray_arg {
            ::size_t size;
            void* ptr;
        };
        
        template <>
        struct KernelArgumentHandler<carray_arg> {
            static ::size_t size(const carray_arg& value) {
                return value.size;
            }
            static void* ptr(carray_arg& value) {
                return value.ptr;
            }
        };
        
    }
}

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
        
        /** Create an argument from c-style array */
        template<class T>
        inline cl::detail::carray_arg carray(const T* ptr, ::size_t n) {
            cl::detail::carray_arg c;
            c.ptr = reinterpret_cast<void*>(const_cast<T*>(ptr));
            c.size = sizeof(T) * n;
            return c;
        }
        
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
            std::string clSourceBake = readFile(std::string(BAKE_PATH) + "/inc/bake/opencl/bake.cl");
            std::string clSourceRay = readFile(std::string(BAKE_PATH) + "/inc/bake/opencl/ray.cl");
            
            cl::Program::Sources sources;
            sources.push_back(std::make_pair(clSourceRay.c_str(), clSourceRay.size()));
            sources.push_back(std::make_pair(clSourceBake.c_str(), clSourceBake.size()));
            
            
            c.prg = cl::Program(c.ctx, sources, &err);
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
        
        bool bakeTextureMap(const Surface &src, const Surface &target) {
            OCL ocl;
            if (!initOpenCL(ocl, 2)) {
                BAKE_LOG("Failed to initialize OpenCL.");
                return false;
            }
            
            SurfaceVolume sv;
            if (!buildSurfaceVolume(src, Eigen::Vector3i::Constant(64), sv)) {
                BAKE_LOG("Failed to create surface volume.");
                return false;
            }
            
            
            // Target
            
            cl_int err;
            
            cl::Buffer bTargetVertexPositions(ocl.ctx,
                                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              target.vertexPositions.array().size() * sizeof(float),
                                              const_cast<float*>(target.vertexPositions.data()),
                                              &err);
            ASSERT_OPENCL(err, "Failed to create vertex buffer for target.");
            
            
            cl::Buffer bTargetVertexUVs(ocl.ctx,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        target.vertexUVs.array().size() * sizeof(float),
                                        const_cast<float*>(target.vertexUVs.data()), &err);
            ASSERT_OPENCL(err, "Failed to create UV buffer for target.");
            
            cl::Buffer bTargetVertexNormals(ocl.ctx,
                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            target.vertexNormals.array().size() * sizeof(float),
                                            const_cast<float*>(target.vertexNormals.data()), &err);
            ASSERT_OPENCL(err, "Failed to create normals buffer for target.");
            
            // Source
            
            cl::Buffer bSrcVertexPositions(ocl.ctx,
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           src.vertexPositions.array().size() * sizeof(float),
                                           const_cast<float*>(src.vertexPositions.data()),
                                           &err);
            ASSERT_OPENCL(err, "Failed to create vertex buffer for source.");
            
            cl::Buffer bSrcVertexNormals(ocl.ctx,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         src.vertexNormals.array().size() * sizeof(float),
                                         const_cast<float*>(src.vertexNormals.data()), &err);
            ASSERT_OPENCL(err, "Failed to create normals buffer for source.");
            
            
            cl::Buffer bSrcVertexColors(ocl.ctx,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        src.vertexColors.array().size() * sizeof(float),
                                        const_cast<float*>(src.vertexColors.data()), &err);
            ASSERT_OPENCL(err, "Failed to create color buffer for source.");
            
            // Volume
            
            cl::Buffer bSrcVoxels(ocl.ctx,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  (int)sv.cells.size() * sizeof(int),
                                  const_cast<int*>(sv.cells.data()),
                                  &err);
            ASSERT_OPENCL(err, "Failed to create voxel buffer for source.");
            
            cl::Buffer bSrcTrianglesInVoxels(ocl.ctx,
                                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             (int)sv.triangleIndices.size() * sizeof(int),
                                             const_cast<int*>(sv.triangleIndices.data()),
                                             &err);
            ASSERT_OPENCL(err, "Failed to triangle index buffer for source.");
            
            float minmax[8] = {
                sv.bounds.min().x(), sv.bounds.min().y(), sv.bounds.min().z(), 0.f,
                sv.bounds.max().x(), sv.bounds.max().y(), sv.bounds.max().z(), 0.f,
            };
            
            cl_float4 voxelSizes = {{sv.voxelSizes.x(), sv.voxelSizes.y(), sv.voxelSizes.z(), 0}};
            cl_int4 voxelsPerDim = {{sv.voxelsPerDimension.x(), sv.voxelsPerDimension.y(), sv.voxelsPerDimension.z(), 0}};
            
            // Texture
            
            const int imagesize = 1024;
            
            Image<unsigned char> texture(imagesize, imagesize, 3);
            texture.toOpenCV().setTo(0);
            
            cl::Image2D bTexture(ocl.ctx,
                                 CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cl::ImageFormat(CL_RGB, CL_UNORM_INT8),
                                 imagesize, imagesize, 0, texture.row(0), &err);
            ASSERT_OPENCL(err, "Failed to create texture image.");
            
            int argc = 0;
            ocl.kBakeTexture.setArg(0, bTargetVertexPositions);
            ocl.kBakeTexture.setArg(1, bTargetVertexNormals);
            ocl.kBakeTexture.setArg(2, bTargetVertexUVs);
            ocl.kBakeTexture.setArg(3, bSrcVertexPositions);
            ocl.kBakeTexture.setArg(4, bSrcVertexNormals);
            ocl.kBakeTexture.setArg(5, bSrcVertexColors);
            ocl.kBakeTexture.setArg(6, bSrcVoxels);
            ocl.kBakeTexture.setArg(7, bSrcTrianglesInVoxels);
            ocl.kBakeTexture.setArg(8, carray(minmax, 8));
            ocl.kBakeTexture.setArg(9, sizeof(cl_float4), voxelSizes.s);
            ocl.kBakeTexture.setArg(10, sizeof(cl_int4), voxelsPerDim.s);
            ocl.kBakeTexture.setArg(11, bTexture);
            ocl.kBakeTexture.setArg(12, imagesize);
            ocl.kBakeTexture.setArg(13, 0.5f);
            ocl.kBakeTexture.setArg(14, (int)target.vertexPositions.cols()/3);
            
            int nTrianglesDivisableBy2 = target.vertexPositions.cols()/3 + (target.vertexPositions.cols()/3) % 2;
            
            err = ocl.q.enqueueNDRangeKernel(ocl.kBakeTexture, cl::NullRange, cl::NDRange(nTrianglesDivisableBy2), cl::NullRange);
            ASSERT_OPENCL(err, "Failed to run bake kernel.");
            
            cl::size_t<3> origin;
            origin.push_back(0);
            origin.push_back(0);
            origin.push_back(0);
            
            cl::size_t<3> region;
            region.push_back(imagesize);
            region.push_back(imagesize);
            region.push_back(1);
            
            err = ocl.q.enqueueReadImage(bTexture, false, origin, region, 0, 0, texture.row(0));
            ASSERT_OPENCL(err, "Failed to read image.");
            ocl.q.finish();
            
            cv::Mat m = texture.toOpenCV();
            cv::flip(m, m, 0);
            
            cv::imwrite("input.png", m);
            cv::imshow("test", m);
            cv::waitKey();
            
            
            return false;
        }
        
    }
}