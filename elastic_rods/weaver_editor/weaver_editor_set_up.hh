#ifndef WEAVEREDITORSETUP_HH
#define WEAVEREDITORSETUP_HH

#include "../SurfaceAttractedLinkage.hh"
#include "../design_parameter_solve.hh"
#include "../infer_target_surface.hh"
#include "../open_linkage.hh"
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/unused.hh>
#include <string>
#include <iostream>
#include <map>
#include <thread>
using Linkage_dPC = DesignParameterConfig;
template<typename Object>
void setUpLinkageForOptimization(Object &l,  double linkage_attraction_weight, bool holdClosestPointsFixed, std::string cross_section_path, bool useFixedJoint) {
    l.set_attraction_tgt_joint_weight(0.01);
    l.attraction_weight = linkage_attraction_weight;
    l.set_holdClosestPointsFixed(holdClosestPointsFixed);
    // TODO: make a custom cross-section that does the same thing as RodMaterial::setContour
    {
        auto ext = cross_section_path.substr(cross_section_path.size() - 4);
        if ((ext == ".obj") || ext == ".msh") {
            RodMaterial mat;
            mat.setContour(20000, 0.3, cross_section_path, 1.0, RodMaterial::StiffAxis::D1,
                    false, "", 0.001, 10);
            l.setMaterial(mat);
        }
        else l.setMaterial(RodMaterial(*CrossSection::load(cross_section_path)));
    }

    std::cout << "Initial design parameter solve:" << std::endl;
    {
        NewtonOptimizerOptions designParameter_eopts;
        designParameter_eopts.niter = 1000;
        designParameter_eopts.verbose = 10;
        l.setDesignParameterConfig(true, true);
        designParameter_solve(l, designParameter_eopts, 0.0, 0.001);
    }

    NewtonOptimizerOptions eopts;
    eopts.gradTol = 1e-6;
    eopts.verbose = 10;
    eopts.beta = 1e-8;
    eopts.niter = 100;

    size_t pinJoint = 0;
    std::cout << "Initial compute equilibrium:" << std::endl;
    {
        const size_t jdo = l.dofOffsetForJoint(pinJoint);
        std::vector<size_t> rigidMotionPinVars;
        for (size_t i = 0; i < 6; ++i) rigidMotionPinVars.push_back(jdo + i);
        if (useFixedJoint)
            compute_equilibrium(l, eopts, rigidMotionPinVars);
        else
            compute_equilibrium(l, eopts);
    }

}
#endif /* end of include guard: WEAVEREDITORSETUP_HH */
