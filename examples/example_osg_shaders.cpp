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

TEST_CASE("osg_shaders")
{
    
    osg::Node *root = osgDB::readNodeFile(std::string("input.obj"));
    
    bake::Surface s;
    bake::convertSurface(root, s, bake::ConvertAll);
    
    /*
    osgViewer::Viewer viewer;
    viewer.setSceneData(root);
    viewer.setCameraManipulator(new osgGA::TrackballManipulator);
    viewer.setUpViewInWindow(200, 200, 640, 480);
    viewer.run();
     */
    
}