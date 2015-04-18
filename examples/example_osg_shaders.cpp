// This file is part of gpu-bake, a library for baking texture maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include "catch.hpp"

#include <osg/Node>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <iostream>

#include <bake/convert_surface.h>
#include <bake/opencl/bake.h>

TEST_CASE("osg_shaders")
{
    
    osgDB::Options *opts = new osgDB::Options();
    opts->setOptionString("noTesselateLargePolygons noTriStripPolygons noRotation");
    
    osg::Node *nSrc = osgDB::readNodeFile(std::string("source.ply"), opts);
    osg::Node *nTarget = osgDB::readNodeFile(std::string("target.obj"), opts);
     
    bake::Surface src, target;
    
    bake::convertSurface(nSrc, src, bake::ConvertVertexNormals | bake::ConvertVertexColors);
    bake::convertSurface(nTarget, target, bake::ConvertVertexNormals | bake::ConvertVertexUVs);
    
    bake::opencl::bakeTextureMap(src, target);
    /*
    osgViewer::Viewer viewer;
    viewer.setSceneData(root);
    viewer.setCameraManipulator(new osgGA::TrackballManipulator);
    viewer.setUpViewInWindow(200, 200, 640, 480);
    viewer.run();
     */
    
}