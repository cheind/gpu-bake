// This file is part of gpu-bake, a library for baking specific maps on GPUs.
//
// Copyright (C) 2015 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the BSD was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <bake/convert_surface.h>
#include <bake/log.h>
#include <osgUtil/Optimizer>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PrimitiveSet>

namespace bake {
    
    /** Runs through the graph and collects element counts and other properties. */
    class FirstPassVisitor : public osg::NodeVisitor {
    public:
        int nTriangles;
        bool hasOnlyTriangles;
        bool hasVertexColors;
        bool hasVertexNormals;
        bool hasVertexUVs;
        
        
        FirstPassVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
        {
            nTriangles = 0;
            hasOnlyTriangles = true;
            hasVertexColors = true;
            hasVertexNormals = true;
            hasVertexUVs = true;
        }
        
        virtual void apply(osg::Geode &n) {
            
            unsigned int nDraws = n.getNumDrawables();
            for (unsigned int idraw = 0 ; idraw < nDraws; ++idraw) {
                osg::Geometry *geom = n.getDrawable( idraw )->asGeometry();
                if (geom) {
                    hasVertexColors &= (geom->getColorArray() != 0) && (geom->getColorBinding() == osg::Geometry::BIND_PER_VERTEX);
                    hasVertexNormals &= (geom->getNormalArray() != 0) && (geom->getNormalBinding() == osg::Geometry::BIND_PER_VERTEX);
                    hasVertexUVs &= (geom->getTexCoordArray(0) != 0);
                    
                    unsigned int nPrims = geom->getNumPrimitiveSets();
                    for (unsigned int iprim = 0; iprim < nPrims; ++iprim) {
                        osg::PrimitiveSet *p = geom->getPrimitiveSet(iprim);
                        if (p->getMode() == osg::PrimitiveSet::TRIANGLES) {
                            nTriangles += p->getNumPrimitives();
                        } else {
                            hasOnlyTriangles = false;
                        }
                    }
                }
            }
            
            traverse(n);
        }
    };
    
    /** Runs through the graph and extracts vertex positions and properties */
    class SecondPassVisitor : public osg::NodeVisitor {
    public:
        
        
        SecondPassVisitor(Surface &s, unsigned int opts)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN), _s(s), _opts(opts), _idx(0)
        {}
        
        virtual void apply(osg::Geode &n) {
            unsigned int nDraws = n.getNumDrawables();
            for (unsigned int idraw = 0 ; idraw < nDraws; ++idraw) {
                osg::Geometry *geom = n.getDrawable( idraw )->asGeometry();
                osg::Vec3Array *v = static_cast<osg::Vec3Array*>(geom->getVertexArray());
                osg::Vec3Array *vc = static_cast<osg::Vec3Array*>(geom->getColorArray());
                osg::Vec3Array *vn = static_cast<osg::Vec3Array*>(geom->getNormalArray());
                osg::Vec2Array *vt = static_cast<osg::Vec2Array*>(geom->getTexCoordArray(0));
                
                if (geom) {
                    unsigned int nPrims = geom->getNumPrimitiveSets();
                    for (unsigned int iprim = 0; iprim < nPrims; ++iprim) {
                        osg::PrimitiveSet *p = geom->getPrimitiveSet(iprim);
                        //osg::DrawArrays *a = dynamic_cast<osg::DrawArrays*>(p->getDrawElements())
                        
                        for (GLsizei i = 0; i < p->getNumIndices(); ++i) {
                            _s.vertexPositions.col(_idx) = toE(v->at(p->index(i)));
                            if (_opts & ConvertVertexColors)
                                _s.vertexColors.col(_idx) = toE(vc->at(p->index(i)));
                            if (_opts & ConvertVertexNormals)
                                _s.vertexNormals.col(_idx) = toE(vn->at(p->index(i)));
                            if (_opts & ConvertVertexColors)
                                _s.vertexUVs.col(_idx) = toE(vt->at(p->index(i)));
                            ++_idx;
                        }
                    }
                }
            }
            traverse(n);
        }
        
    private:
        
        Eigen::Vector3f toE(const osg::Vec3 &v) const {
            return Eigen::Vector3f(v.x(), v.y(), v.z());
        }
        
        Eigen::Vector2f toE(const osg::Vec2 &v) const {
            return Eigen::Vector2f(v.x(), v.y());
        }
        
        Surface &_s;
        unsigned int _opts;
        Surface::VertexPositionMatrix::Index _idx;
    };

    
    
    bool convertSurface(const osg::Node *node, Surface &s, unsigned int opts)
    {
        osgUtil::Optimizer opt;
        osg::ref_ptr<osg::Node> n = (osg::Node*)node->clone(osg::CopyOp::DEEP_COPY_ALL);
        opt.optimize(n,
                     osgUtil::Optimizer::MERGE_GEODES |
                     osgUtil::Optimizer::MERGE_GEOMETRY |
                     osgUtil::Optimizer::INDEX_MESH);
        
        FirstPassVisitor v1;
        n->accept(v1);
        
        // Check constraints
        
        if (v1.nTriangles == 0) {
            BAKE_LOG("Zero triangles found.");
            return false;
        } else {
            BAKE_LOG("Found %d triangles", v1.nTriangles);
        }
        
        if ((opts & ConvertVertexColors) && !v1.hasVertexColors) {
            BAKE_LOG("Vertex colors requested but not found.");
            return false;
        }
        
        if ((opts & ConvertVertexNormals) && !v1.hasVertexNormals) {
            BAKE_LOG("Vertex normals requested but not found.");
            return false;
        }
        
        if ((opts & ConvertVertexUVs) && !v1.hasVertexUVs) {
            BAKE_LOG("Vertex UVs requested but not found.");
            return false;
        }

        // Alloc storage
        s.vertexPositions.resize(3, v1.nTriangles * 3);
        if (opts & ConvertVertexColors) s.vertexColors.resize(3, v1.nTriangles * 3);
        if (opts & ConvertVertexNormals) s.vertexNormals.resize(3, v1.nTriangles * 3);
        if (opts & ConvertVertexUVs) s.vertexUVs.resize(2, v1.nTriangles * 3);
        
        SecondPassVisitor v2(s, opts);
        return true;
        
    }
                     
    
}