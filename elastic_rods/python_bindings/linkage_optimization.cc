#include <MeshFEM/Geometry.hh>

#include "../SurfaceAttractedLinkage.hh"
#include "../DesignOptimizationTerms.hh"
#include "../TargetSurfaceFitterMesh.hh"

#include "../RegularizationTerms.hh"
#include "../LinkageOptimization.hh"
#include "../WeavingOptimization.hh"
#include "../XShellOptimization.hh"
#include "../CShellOptimization.hh"
#include "../ConstrainedCShellOptimization.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include <sstream>
namespace py = pybind11;

template<typename Real_>
using SAL_T = SurfaceAttractedLinkage_T<Real_>;
using SAL   = SAL_T<Real>;

template<typename T>
std::string hexString(T val) {
    std::ostringstream ss;
    ss << std::hex << val;
    return ss.str();
}

template<template<typename> class Object>
struct LinkageOptimizationTrampoline : public LinkageOptimization<Object> {
    // Inherit the constructors.
    using LinkageOptimization<Object>::LinkageOptimization;

    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::VectorXd, // Return type.
            LinkageOptimization<Object>, // Parent class.
            gradp_J, // Name of the function in C++.
            params, opt_eType// Arguments.
        );
    }

    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c = 0.0, Real coeff_angle_constraint = 0.0, OptEnergyType opt_eType = OptEnergyType::Full) override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::VectorXd, // Return type.
            LinkageOptimization<Object>, // Parent class.
            apply_hess, // Name of the function in C++.
            params, delta_p, coeff_J, coeff_c, coeff_angle_constraint, opt_eType// Arguments.
        );
    }
    void setLinkageInterleavingType(InterleavingType new_type) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setLinkageInterleavingType, // Name of the function in C++.
            new_type// Arguments.
        );
    }
    void commitLinesearchLinkage() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            commitLinesearchLinkage, // Name of the function in C++.
            // No Arguments.
        );
    }
    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setEquilibriumOptions, // Name of the function in C++.
            eopts// Arguments.
        );
    }
    NewtonOptimizerOptions getEquilibriumOptions() const override {
        PYBIND11_OVERRIDE_PURE(
            NewtonOptimizerOptions, // Return type.
            LinkageOptimization<Object>, // Parent class.
            getEquilibriumOptions, // Name of the function in C++.
            // No Arguments.
        );
    }
    void setGamma(Real val) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setGamma, // Name of the function in C++.
            val// Arguments.
        );
    }
    Real getGamma() const override {
        PYBIND11_OVERRIDE_PURE(
            Real, // Return type.
            LinkageOptimization<Object>, // Parent class.
            getGamma, // Name of the function in C++.
            // No Arguments.
        );
    }

    void m_forceEquilibriumUpdate() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_forceEquilibriumUpdate, // Name of the function in C++.
            // No Arguments.
        );
    }
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params) override {
        PYBIND11_OVERRIDE_PURE(
            bool, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateEquilibria, // Name of the function in C++.
            params// Arguments.
        );
    }
    void m_updateClosestPoints() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateClosestPoints, // Name of the function in C++.
            // No Arguments.
        );
    }
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params) override {
        PYBIND11_OVERRIDE_PURE(
            bool, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateAdjointState, // Name of the function in C++.
            params// Arguments.
        );
    }

};

template<template<typename> class Object>
void bindLinkageOptimization(py::module &m, const std::string &typestr) {
    using LO = LinkageOptimization<Object>;
    using LTO = LinkageOptimizationTrampoline<Object>;
    std::string pyclass_name = std::string("LinkageOptimization_") + typestr;
    py::class_<LO, LTO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, const NewtonOptimizerOptions &, Real, Real, Real>(), py::arg("baseLinkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("E0") = 1.0, py::arg("l0") = 1.0, py::arg("rl0") = 1.0)
    .def("newPt",          &LO::newPt, py::arg("params"))
    .def("params",         &LO::params)
    .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&LO::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("J_target",       py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LO::J_target),       py::arg("params"))
    .def("J_regularization", &LO::J_regularization)
    .def("J_smoothing",      &LO::J_smoothing)
    .def("apply_hess_J",   &LO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
    .def("apply_hess_c",   &LO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
    .def("numParams",       &LO::numParams)
    .def("get_rl0",         &LO::get_rl0)
    .def("get_E0",          &LO::get_E0)
    .def("invalidateAdjointState",     &LO::invalidateAdjointState)
    .def("restKappaSmoothness", &LO::restKappaSmoothness)
    .def_readwrite("prediction_order", &LO::prediction_order)
    .def_property("beta",  &LO::getBeta , &LO::setBeta )
    .def_property("gamma", &LO::getGamma, &LO::setGamma)
    .def_property("rl_regularization_weight", &LO::getRestLengthMinimizationWeight, &LO::setRestLengthMinimizationWeight)
    .def_property("smoothing_weight",         &LO::getRestKappaSmoothingWeight,     &LO::setRestKappaSmoothingWeight)
    .def_property("contact_force_weight",     &LO::getContactForceWeight,           &LO::setContactForceWeight)
    .def_readonly("target_surface_fitter",    &LO::target_surface_fitter)
    .def_readwrite("objective", &LO::objective, py::return_value_policy::reference)
    ;
}

template<template<typename> class Object>
void bindXShellOptimization(py::module &m, const std::string &typestr) {
    using XO = XShellOptimization<Object>;
    using LO = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("XShellOptimization_") + typestr;
    py::class_<XO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, Object<Real> &, const NewtonOptimizerOptions &, Real, int, bool>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true)
    .def("c",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::c),              py::arg("params"))
    .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&XO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("gradp_J_target", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::gradp_J_target), py::arg("params"))
    .def("gradp_c",        &XO::gradp_c,        py::arg("params"))
    .def("get_w_x",       &XO::get_w_x)
    .def("get_y",         &XO::get_y)

    .def("get_s_x",       &XO::get_s_x)
    .def("get_delta_x3d", &XO::get_delta_x3d)
    .def("get_delta_x2d", &XO::get_delta_x2d)
    .def("get_delta_w_x", &XO::get_delta_w_x)
    .def("get_delta_s_x", &XO::get_delta_s_x)

    .def("pushforward", &XO::pushforward, py::arg("params"), py::arg("delta_p"))
    .def("apply_hess",  &XO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J") = 1., py::arg("coeff_c") = 0., py::arg("coeff_angle_constraint") = 0., py::arg("energyType") = OptEnergyType::Full)
    .def("loadTargetSurface",          &XO::loadTargetSurface,          py::arg("path"))
    .def("getLinesearchBaseLinkage",     &XO::getLinesearchBaseLinkage,     py::return_value_policy::reference)
    .def("getLinesearchDeployedLinkage", &XO::getLinesearchDeployedLinkage, py::return_value_policy::reference)
    .def("constructTargetSurface", &XO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("scale_factors"))
    .def("XShellOptimize", &XO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = true, py::arg("applyFlatnessConstraint") = true)
    ;
}

template<template<typename> class Object>
void bindCShellOptimization(py::module &m, const std::string &typestr) {
    using CO = CShellOptimization<Object>;
    using LO = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("CShellOptimization_") + typestr;
    py::class_<CO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, Object<Real> &, const NewtonOptimizerOptions &, Real, int, bool>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true)
    .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&CO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("gradp_J_target", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&CO::gradp_J_target), py::arg("params"))
    .def("get_w_x",       &CO::get_w_x)

    .def("get_delta_x3d", &CO::get_delta_x3d)
    .def("get_delta_w_x", &CO::get_delta_w_x)

    .def("pushforward", &CO::pushforward, py::arg("params"), py::arg("delta_p"))
    .def("apply_hess",  &CO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J") = 1., py::arg("coeff_c") = 0., py::arg("coeff_angle_constraint") = 0., py::arg("energyType") = OptEnergyType::Full)
    .def("getLinesearchBaseLinkage",     &CO::getLinesearchBaseLinkage,     py::return_value_policy::reference)
    .def("getLinesearchDeployedLinkage", &CO::getLinesearchDeployedLinkage, py::return_value_policy::reference)
    .def("constructTargetSurface", &CO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("scale_factors"))
    .def("CShellOptimize", &CO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = true, py::arg("applyFlatnessConstraint") = true)
    ;
}

template<template<typename> class Object>
void bindConstrainedCShellOptimization(py::module &m, const std::string &typestr) {
    using CCO = ConstrainedCShellOptimization<Object>;
    using LO  = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("ConstrainedCShellOptimization_") + typestr;
    py::class_<CCO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, Object<Real> &, const NewtonOptimizerOptions &, Real, int, bool>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true)
    .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&CCO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("gradp_J_target", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&CCO::gradp_J_target), py::arg("params"))
    .def("get_w_x",       &CCO::get_w_x)

    .def("get_delta_x3d", &CCO::get_delta_x3d)
    .def("get_delta_w_x", &CCO::get_delta_w_x)

    .def("pushforward", &CCO::pushforward, py::arg("params"), py::arg("delta_p"))
    .def("apply_hess",  &CCO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J") = 1., py::arg("coeff_c") = 0., py::arg("coeff_angle_constraint") = 0., py::arg("energyType") = OptEnergyType::Full)
    .def("getLinesearchBaseLinkage",     &CCO::getLinesearchBaseLinkage,     py::return_value_policy::reference)
    .def("getLinesearchDeployedLinkage", &CCO::getLinesearchDeployedLinkage, py::return_value_policy::reference)
    .def("constructTargetSurface", &CCO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("scale_factors"))
    .def("CShellOptimize", &CCO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = true, py::arg("applyFlatnessConstraint") = true)
    ;
}

template<template<typename> class Object>
void bindWeavingOptimization(py::module &m, const std::string &typestr) {
    using WO = WeavingOptimization<Object>;
    using LO = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("WeavingOptimization_") + typestr;
    py::class_<WO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, const std::string, bool, const NewtonOptimizerOptions &, int, bool, const std::vector<size_t>>(), py::arg("weaver"), py::arg("input_surface_path"), py::arg("useCenterline"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("pinJoint") = -1, py::arg("useFixedJoint") = true, py::arg("fixedVars") = std::vector<size_t>())
    .def("gradp_J",        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&WO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)

    .def("WeavingOptimize", &WO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = false, py::arg("applyFlatnessConstraint") = false)
    .def("get_w_x",         &WO::get_w_x)

    .def("get_delta_x",   &WO::get_delta_x)
    .def("get_delta_w_x", &WO::get_delta_w_x)

    .def("getLinesearchWeaverLinkage", &WO::getLinesearchWeaverLinkage, py::return_value_policy::reference)
    .def("setWeavingOptimization",     &WO::setWeavingOptimization)
    .def("setUseCenterline",           &WO::setUseCenterline, py::arg("useCenterline"), py::arg("jointPosWeight"), py::arg("jointPosValence2Multiplier"))

    .def("loadTargetSurface",          &WO::loadTargetSurface,          py::arg("path"))
    .def("set_target_joint_position",  &WO::set_target_joint_position,  py::arg("input_target_joint_pos"))
    .def("set_holdClosestPointsFixed", &WO::set_holdClosestPointsFixed, py::arg("holdClosestPointsFixed"))
    .def("scaleJointWeights",          &WO::scaleJointWeights, py::arg("jointPosWeight"), py::arg("featureMultiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
    .def("get_holdClosestPointsFixed", &WO::get_holdClosestPointsFixed)
    .def("setLinkageAttractionWeight", &WO::setLinkageAttractionWeight, py::arg("attraction_weight"))
    .def("constructTargetSurface", &WO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("scale_factors"))
    .def("equilibriumOptions",  &WO::equilibriumOptions,  py::return_value_policy::reference)
    ;
}

PYBIND11_MODULE(linkage_optimization, m) {
    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("elastic_rods");
    m.doc() = "Linkage Optimization Codebase";

    py::module detail_module = m.def_submodule("detail");

    py::enum_<OptEnergyType>(m, "OptEnergyType")
        .value("Full",           OptEnergyType::Full           )
        .value("ElasticBase",    OptEnergyType::ElasticBase    )
        .value("ElasticDeployed",OptEnergyType::ElasticDeployed)
        .value("Target",         OptEnergyType::Target         )
        .value("Regularization", OptEnergyType::Regularization )
        .value("Smoothing",      OptEnergyType::Smoothing      )
        .value("ContactForce",   OptEnergyType::ContactForce   )
        ;

    using TSF = TargetSurfaceFitter;
    py::class_<TSF>(m, "TargetSurfaceFitter")
        .def(py::init<>())
        .def("loadTargetSurface", &TSF::loadTargetSurface, py::arg("linkage"), py::arg("surf_path"))
        .def("objective",       &TSF::objective,             py::arg("linkage"))
        .def("gradient",        &TSF::gradient,              py::arg("linkage"))
        .def("numSamplePoints", &TSF::numSamplePoints<Real>, py::arg("linkage"))

        .def("setTargetJointPosVsTargetSurfaceTradeoff", &TSF::setTargetJointPosVsTargetSurfaceTradeoff<Real>, py::arg("linkage"), py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0)
        .def("scaleJointWeights", &TSF::scaleJointWeights<Real>,  py::arg("linkage"), py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
        .def_readwrite("holdClosestPointsFixed", &TSF::holdClosestPointsFixed)

        .def_readonly("W_diag_joint_pos",                      &TSF::W_diag_joint_pos)
        .def_readonly("Wsurf_diag_linkage_sample_pos",         &TSF::Wsurf_diag_linkage_sample_pos)
        .def_readonly("joint_pos_tgt",                         &TSF::joint_pos_tgt)

        .def_property_readonly("V", [](const TSF &tsf) { return tsf.getV(); })
        .def_property_readonly("F", [](const TSF &tsf) { return tsf.getF(); })
        .def_property_readonly("N", [](const TSF &tsf) { return tsf.getN(); })
        .def_readonly("linkage_closest_surf_pts",              &TSF::linkage_closest_surf_pts)
        .def_readonly("linkage_closest_surf_pt_sensitivities", &TSF::linkage_closest_surf_pt_sensitivities)
        .def_readonly("linkage_closest_surf_tris",             &TSF::linkage_closest_surf_tris)
        .def_readonly("holdClosestPointsFixed",                &TSF::holdClosestPointsFixed)
        ;

    using RT = RegularizationTerm<SAL>;
    py::class_<RT, std::shared_ptr<RT>>(m, "RegularizationTerm")
        .def("energy", &RT::energy)
        .def_readwrite("weight", &RT::weight)
        ;

    using RCS = RestCurvatureSmoothing<SAL>;
    py::class_<RCS, RT, std::shared_ptr<RCS>>(m, "RestCurvatureSmoothing")
        .def(py::init<const SAL &>(), py::arg("linkage"))
        ;

    using RLM = RestLengthMinimization<SAL>;
    py::class_<RLM, RT, std::shared_ptr<RLM>>(m, "RestLengthMinimization")
        .def(py::init<const SAL &>(), py::arg("linkage"))
        ;

    using DOT = DesignOptimizationTerm<SAL_T>;
    py::class_<DOT, std::shared_ptr<DOT>>(m, "DesignOptimizationTerm")
        .def("value",  &DOT::value)
        .def("update", &DOT::update)
        .def("grad"  , &DOT::grad  )
        .def("grad_x", &DOT::grad_x)
        .def("grad_p", &DOT::grad_p)
        .def("computeGrad",      &DOT::computeGrad)
        .def("computeDeltaGrad", &DOT::computeDeltaGrad, py::arg("delta_xp"))
        ;

    using DOOT = DesignOptimizationObjectiveTerm<SAL_T>;
    py::class_<DOOT, DOT, std::shared_ptr<DOOT>>(m, "DesignOptimizationObjectiveTerm")
        .def_readwrite("weight", &DOOT::weight)
        ;

    using EEO = ElasticEnergyObjective<SAL_T>;
    py::class_<EEO, DOOT, std::shared_ptr<EEO>>(m, "ElasticEnergyObjective")
        .def(py::init<const SAL &>(), py::arg("surface_attracted_linkage"))
        .def_property("useEnvelopeTheorem", &EEO::useEnvelopeTheorem, &EEO::setUseEnvelopeTheorem)
        ;

    using CFO = ContactForceObjective<SAL_T>;
    py::class_<CFO, DOOT, std::shared_ptr<CFO>>(m, "ContactForceObjective")
        .def(py::init<const SAL &>(), py::arg("surface_attracted_linkage"))
        .def_property(             "normalWeight", &CFO::getNormalWeight,              &CFO::setNormalWeight)
        .def_property(         "tangentialWeight", &CFO::getTangentialWeight,          &CFO::setTangentialWeight)
        .def_property(             "torqueWeight", &CFO::getTorqueWeight,          &CFO::setTorqueWeight)
        .def_property(     "boundaryNormalWeight", &CFO::getBoundaryNormalWeight,      &CFO::setBoundaryNormalWeight)
        .def_property( "boundaryTangentialWeight", &CFO::getBoundaryTangentialWeight,  &CFO::setBoundaryTangentialWeight)
        .def_property(     "boundaryTorqueWeight", &CFO::getBoundaryTorqueWeight,  &CFO::setBoundaryTorqueWeight)
        .def_property("normalActivationThreshold", &CFO::getNormalActivationThreshold, &CFO::setNormalActivationThreshold)
        .def("jointForces", [](const CFO &cfo) { return cfo.jointForces(); })
        ;

    using TFO = TargetFittingDOOT<SAL_T>;
    py::class_<TFO, DOOT, std::shared_ptr<TFO>>(m, "TargetFittingDOOT")
        .def(py::init<const SAL &, TargetSurfaceFitter &>(), py::arg("surface_attracted_linkage"), py::arg("targetSurfaceFitter"))
        ;

    using RCSD = RegularizationTermDOOWrapper<SAL_T, RestCurvatureSmoothing>;
    py::class_<RCSD, DOT, std::shared_ptr<RCSD>>(m, "RestCurvatureSmoothingDOOT")
        .def(py::init<const SAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RCS>>(), py::arg("restCurvatureRegTerm"))
        .def_property("weight", [](const RCSD &r) { return r.weight; }, [](RCSD &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RLMD = RegularizationTermDOOWrapper<SAL_T, RestLengthMinimization>;
    py::class_<RLMD, DOT, std::shared_ptr<RLMD>>(m, "RestLengthMinimizationDOOT")
        .def(py::init<const SAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RLM>>(), py::arg("restLengthMinimizationTerm"))
        .def_property("weight", [](const RLMD &r) { return r.weight; }, [](RLMD &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using OEType = OptEnergyType;
    using DOO  = DesignOptimizationObjective<SAL_T, OEType>;
    using TR   = DOO::TermRecord;
    using TPtr = DOO::TermPtr;
    py::class_<DOO> doo(m, "DesignOptimizationObjective");

    py::class_<TR>(doo, "DesignOptimizationObjectiveTermRecord")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR::name)
        .def_readwrite("type", &TR::type)
        .def_readwrite("term", &TR::term)
        .def("__repr__", [](const TR *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo.def(py::init<>())
       .def("update",         &DOO::update)
       .def("grad",           &DOO::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO::values)
       .def("weightedValues", &DOO::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO::value, py::const_), py::arg("type") = OEType::Full)
        .def("computeGrad",     &DOO::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr      >(&DOO::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr, Real>(&DOO::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO &o, const std::list<std::tuple<std::string, OEType, TPtr>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;

    ////////////////////////////////////////////////////////////////////////////////
    // Linkage Optimization Base Class
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<OptAlgorithm>(m, "OptAlgorithm")
        .value("NEWTON_CG", OptAlgorithm::NEWTON_CG)
        .value("BFGS",      OptAlgorithm::BFGS     )
        ;


    bindLinkageOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    bindLinkageOptimization<SAL_T>(detail_module, "SurfaceAttractedLinkage");
    ////////////////////////////////////////////////////////////////////////////////
    // XShell Optimization
    ////////////////////////////////////////////////////////////////////////////////
    bindXShellOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    m.def("XShellOptimization", [](RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation) {
          // Uncomment this part once XShell Optimization is tested with Surface Attraction Linkage.
          // // Note: while py::cast is not yet documented in the official documentation,
          // // it accepts the return_value_policy as discussed in:
          // //      https://github.com/pybind/pybind11/issues/1201
          // // by setting the return value policy to take_ownership, we can avoid
          // // memory leaks and double frees regardless of the holder type for XShellOptimization_*.
          // try {
          //   auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
          //   return py::cast(new XShellOptimization<SAL_T>(sl, input_surface_path, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          // }
          // catch (...) {
          return py::cast(new XShellOptimization<RodLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation), py::return_value_policy::take_ownership);
          // }
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true);

    ////////////////////////////////////////////////////////////////////////////////
    // CShell Optimization
    ////////////////////////////////////////////////////////////////////////////////
    bindCShellOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    m.def("CShellOptimization", [](RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation) {
          return py::cast(new CShellOptimization<RodLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation), py::return_value_policy::take_ownership);
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true);
    // bindCShellOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    // m.def("CShellOptimization", [](RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation) {
    //       return py::cast(new CShellOptimization<RodLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation), py::return_value_policy::take_ownership);
    // }, py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true);

    bindConstrainedCShellOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    m.def("ConstrainedCShellOptimization", [](RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation) {
          return py::cast(new ConstrainedCShellOptimization<RodLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation), py::return_value_policy::take_ownership);
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true);

    ////////////////////////////////////////////////////////////////////////////////
    // Weaving Optimization
    ////////////////////////////////////////////////////////////////////////////////
    bindWeavingOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    bindWeavingOptimization<SAL_T>(detail_module, "SurfaceAttractedLinkage");
    m.def("WeavingOptimization", [](RodLinkage &linkage, const std::string &input_surface_path, bool useCenterline, const NewtonOptimizerOptions &equilibrium_options, int pinJoint, bool useFixedJoint, const std::vector<size_t> fixedVars) {
          // Note: while py::cast is not yet documented in the official documentation,
          // it accepts the return_value_policy as discussed in:
          //      https://github.com/pybind/pybind11/issues/1201
          // by setting the return value policy to take_ownership, we can avoid
          // memory leaks and double frees regardless of the holder type for WeavingOptimization_*.
          try {
            auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
            return py::cast(new WeavingOptimization<SAL_T>(sl, input_surface_path, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          }
          catch (...) {
            return py::cast(new WeavingOptimization<RodLinkage_T>(linkage, input_surface_path, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          }
    }, py::arg("linkage"), py::arg("input_surface_path"), py::arg("useCenterline"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("pinJoint") = -1, py::arg("useFixedJoint") = true, py::arg("fixedVars") = std::vector<size_t>());

    //bindWeavingOptimization<SAL_T>(detail_module, "SurfaceAttractedLinkage");
    m.def("WeavingOptimization", [](RodLinkage &linkage, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, bool useCenterline, const NewtonOptimizerOptions &equilibrium_options, int pinJoint, bool useFixedJoint, const std::vector<size_t> fixedVars) {
          try {
            auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
            return py::cast(new WeavingOptimization<SAL_T>(sl, V, F, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          }
          catch (...) {
            return py::cast(new WeavingOptimization<RodLinkage_T>(linkage, V, F, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          }
    }, py::arg("linkage"), py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("pinJoint") = -1, py::arg("useFixedJoint") = true, py::arg("fixedVars") = std::vector<size_t>());
    
    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;
}
