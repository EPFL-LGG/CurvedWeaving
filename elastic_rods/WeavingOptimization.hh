////////////////////////////////////////////////////////////////////////////////
// WeavingOptimization.hh
////////////////////////////////////////////////////////////////////////////////

#ifndef WEAVINGOPTIMIZATION_HH
#define WEAVINGOPTIMIZATION_HH

// #include "RodLinkage.hh"
#include "SurfaceAttractedLinkage.hh"
#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "TargetSurfaceFitter.hh"
#include "RegularizationTerms.hh"
#include "LinkageOptimization.hh"

template<template<typename> class Object>
struct WeavingOptimization : public LinkageOptimization<Object>{
    using LO = LinkageOptimization<Object>;
    using LO::m_linesearch_base;
    using LO::m_base;
    using LO::target_surface_fitter;
    using LO::objective;
    using LO::invalidateAdjointState;
    using LO::m_numParams;
    using LO::m_E0;
    using LO::beta;
    using LO::m_l0;
    using LO::m_rigidMotionFixedVars;
    using LO::m_rm_constrained_joint;
    using LO::m_equilibrium_options;
    using LO::m_equilibriumSolveSuccessful;
    using LO::m_adjointStateIsCurrent;
    using LO::m_autodiffLinkagesAreCurrent;
    using LO::prediction_order;
    using LO::numParams;
#if HAS_KNITRO
    using LO::optimize;
#endif

    WeavingOptimization(Object<Real> &weaver, std::string input_surface_path, bool useCenterline, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), int pinJoint = -1, bool useFixedJoint = true, const std::vector<size_t> &fixedVars = std::vector<size_t>());
    WeavingOptimization(Object<Real> &weaver, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, bool useCenterline, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), int pinJoint = -1, bool useFixedJoint = true, const std::vector<size_t> &fixedVars = std::vector<size_t>());

    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full);

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c = 0.0, Real coeff_angle_constraint = 0.0, OptEnergyType opt_eType = OptEnergyType::Full);

    // Access adjoint state for debugging
    Eigen::VectorXd get_w_x() const { return m_w_x; }

    Eigen::VectorXd get_delta_x  () const { return m_delta_x; }
    Eigen::VectorXd get_delta_w_x() const { return m_delta_w_x; }

    Eigen::VectorXd get_delta_delta_x () const { return m_delta_delta_x; }
    Eigen::VectorXd get_second_order_x() const { return m_second_order_x; }

    Object<Real>  get_linesearch_weaver() { return m_linesearch_base; }
    Object<Real> &getLinesearchWeaverLinkage() { return m_linesearch_base; }

    void setLinkageAttractionWeight(Real attraction_weight) {
        // Note: the following will throw a `std::bad_cast` exception if
        // `Object` is not actually a surface attracted linkage.
        auto &saWeave           = dynamic_cast<SurfaceAttractedLinkage_T<  Real> &>(m_base             );
        auto &saLinesearchWeave = dynamic_cast<SurfaceAttractedLinkage_T<  Real> &>(m_linesearch_base  );
        auto &saDiffWeave       = dynamic_cast<SurfaceAttractedLinkage_T<ADReal> &>(m_diff_linkage_weaver);

        Real old_weight = saWeave.attraction_weight;

        saWeave          .attraction_weight = attraction_weight;
        saLinesearchWeave.attraction_weight = attraction_weight;
        saDiffWeave      .attraction_weight = attraction_weight;

        try {
            m_forceEquilibriumUpdate();
            commitLinesearchLinkage();
        }
        catch (std::exception &e) {
            // Roll back in case equilibrium solve failed.
            saWeave          .attraction_weight = old_weight;
            saLinesearchWeave.attraction_weight = old_weight;
            saDiffWeave      .attraction_weight = old_weight;

            throw e;
        }
    }

    void setLinkageInterleavingType(InterleavingType new_type) {
        m_linesearch_base.  set_interleaving_type(new_type);
        m_base.             set_interleaving_type(new_type);
        m_diff_linkage_weaver.set_interleaving_type(new_type);

        // Apply the new joint configuration to the rod segment terminals.
        m_linesearch_base.  setDoFs(  m_linesearch_base.getDoFs(), true /* set spatially coherent thetas */);
        m_base.             setDoFs(             m_base.getDoFs(), true /* set spatially coherent thetas */);
        m_diff_linkage_weaver.setDoFs(m_diff_linkage_weaver.getDoFs(), true /* set spatially coherent thetas */);
    }

    void set_holdClosestPointsFixed(bool holdClosestPointsFixed)           { target_surface_fitter.holdClosestPointsFixed = holdClosestPointsFixed; }
    bool get_holdClosestPointsFixed() const                         { return target_surface_fitter.holdClosestPointsFixed; }
    void set_target_joint_position(Eigen::VectorXd input_target_joint_pos) { target_surface_fitter.joint_pos_tgt = input_target_joint_pos; objective.update(); invalidateAdjointState(); }

    void scaleJointWeights(Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>()) {
        target_surface_fitter.scaleJointWeights(m_linesearch_base, jointPosWeight, featureMultiplier, additional_feature_pts); 
        objective.update(); 
        invalidateAdjointState(); 
    }
    void setWeavingOptimization(Object<Real> &weaver) {
        m_numParams = weaver.numDesignParams();
        m_E0 = weaver.energy();
        m_linesearch_base.set(weaver);
        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void commitLinesearchLinkage() {
        m_base.set(m_linesearch_base);
        // Stash the current factorizations to be reused at each step of the linesearch
        // to predict the equilibrium at the new design parameters.
        getWeaverOptimizer().solver.stashFactorization();
    }

    void setUseCenterline(bool useCenterline, double jointPosWeight, double jointPosValence2Multiplier) {
        target_surface_fitter.setUseCenterline(m_base, useCenterline, jointPosWeight, jointPosValence2Multiplier);
        invalidateAdjointState();
    }

    // Construct/update a target surface for the surface fitting term by
    // inferring a smooth surface from the current joint positions.
    // Also update the closest point projections.
    void constructTargetSurface(size_t loop_subdivisions = 0, Eigen::Vector3d scale_factors = Eigen::Vector3d(1, 1, 1)) {
        target_surface_fitter.constructTargetSurface(m_linesearch_base, loop_subdivisions, scale_factors);
        invalidateAdjointState();
    }

    void loadTargetSurface(const std::string &path) {
        target_surface_fitter.loadTargetSurface(m_base, path);
        invalidateAdjointState();
    }

    void loadTargetSurfaceFromData(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
        target_surface_fitter.setTargetSurface(m_base, V, F);
        invalidateAdjointState();
    }

    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) { getWeaverOptimizer().options = eopts; }
    NewtonOptimizerOptions getEquilibriumOptions() const { return m_weaver_optimizer->options; }
    NewtonOptimizerOptions &equilibriumOptions() { return m_weaver_optimizer->options; }

    NewtonOptimizer &getWeaverOptimizer() { return *m_weaver_optimizer; }

    // Write the full, dense Hessians of J and angle_constraint to a file.
    void dumpHessians(const std::string &/* hess_J_path */, const std::string &/* hess_ac_path */, Real /* fd_eps */ = 1e-5) { throw std::runtime_error("Debug in Python instead..."); }

    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    void setGamma(Real val)                        { objective.get("ElasticEnergy").setWeight(val / m_E0); invalidateAdjointState();}
    Real getGamma()                        const { return objective.get("ElasticEnergy").getWeight() * m_E0; }


private:
    void m_forceEquilibriumUpdate();
    // Return whether "params" are actually new...
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Update the closest point projections for each joint to the target surface.
    void m_updateClosestPoints() { target_surface_fitter.updateClosestPoints(m_linesearch_base); }

    // Update the adjoint state vectors "w" and "y"
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params);

    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXd m_w_x; // adjoint state vectors
    Eigen::VectorXd m_delta_w_x, m_delta_x; // variations of adjoint/forward state from the last call to apply_hess (for debugging)
    Eigen::VectorXd m_delta_delta_x;   // second variations of forward state from last call to m_updateEquilibrium (for debugging)
    Eigen::VectorXd m_second_order_x; // second-order predictions of the linkage's equilibrium (for debugging)
    Eigen::VectorXd m_d3E_w;
    Eigen::VectorXd m_w_rhs, m_delta_w_rhs;
    // Real m_w_lambda, m_delta_w_lambda;

    std::unique_ptr<NewtonOptimizer> m_weaver_optimizer;

    Object<ADReal> m_diff_linkage_weaver;

    RestCurvatureSmoothing<Object<Real>> m_restKappaSmoothing;
    RestLengthMinimization<Object<Real>> m_restLengthMinimization;
};

#include "WeavingOptimization.inl"
#endif /* end of include guard: WEAVINGOPTIMIZATION_HH */
