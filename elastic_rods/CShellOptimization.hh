////////////////////////////////////////////////////////////////////////////////
// CShellOptimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Evaluates objective, constraints, and gradients for solving the following
//  optimal design problem for rod linkages:
//      min_p J(p)

//      J(p) =      1 / E_0 E(x_3D(p), p) +
//             beta / (2 l_0^2) ||x_3D(p) - x_tgt||_W^2
//
//      where x_3D is the equilibrium configuration of the opened linkage,
//            x_tgt are the user-specified target positions for each joint
//            beta, W are weights
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef CSHELLOPTIMIZATION_HH
#define CSHELLOPTIMIZATION_HH

#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"
#include "LinkageOptimization.hh"

template<template<typename> class Object>
struct CShellOptimization : public LinkageOptimization<Object>{
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
    using LO::m_rl0;
    using LO::params;
    using LO::getRestLengthMinimizationWeight;
#if HAS_KNITRO
    using LO::optimize;
#endif
    using LO::apply_hess_J;
    using LO::J;
    // allowFlatActuation: whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.
    CShellOptimization(Object<Real> &flat, Object<Real> &deployed, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), std::unique_ptr<LOMinAngleConstraint<Object>> &&minAngleConstraint = std::unique_ptr<LOMinAngleConstraint<Object>>(), int pinJoint = -1, bool allowFlatActuation = true);

    // It is easier to pybind this constructor since there is no need to bind LOMinAngleConstraint.
    CShellOptimization(Object<Real> &flat, Object<Real> &deployed, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), Real minAngleConstraint = 0 , int pinJoint = -1, bool allowFlatActuation = true) 
        : CShellOptimization(flat, deployed, eopts, std::make_unique<LOMinAngleConstraint<Object>>(minAngleConstraint), pinJoint, allowFlatActuation) {}

    Eigen::VectorXd gradp_J_target() { return gradp_J_target(params()); }

    Real J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
        m_updateEquilibria(params);
        return target_surface_fitter.objective(m_linesearch_deployed);
    }

    Eigen::VectorXd gradp_J               (const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType wopt_eType = OptEnergyType::Full);
    Eigen::VectorXd gradp_J_target        (const Eigen::Ref<const Eigen::VectorXd> &params);
    Eigen::VectorXd pushforward           (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p);


    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c, Real coeff_angle_constraint, OptEnergyType opt_eType = OptEnergyType::Full);

    // Access adjoint state for debugging
    Eigen::VectorXd get_w_x() const { return m_w_x; }
    Eigen::VectorXd get_delta_x3d() const { return m_delta_x3d; }
    Eigen::VectorXd get_delta_w_x() const { return m_delta_w_x; }

    Eigen::VectorXd get_delta_delta_x3d() const { return m_delta_delta_x3d; }
    Eigen::VectorXd get_second_order_x3d() const { return m_second_order_x3d; }

    // The base linkage from the parent class is the flat linkage.
    Object<Real> &getLinesearchFlatLinkage()     { return m_linesearch_base; }

    Object<Real> &getLinesearchDeployedLinkage() { return m_linesearch_deployed; }

    void setLinkageInterleavingType(InterleavingType new_type) {
        m_linesearch_base.  set_interleaving_type(new_type);
        m_base.             set_interleaving_type(new_type);
        // m_diff_linkage_flat.set_interleaving_type(new_type);

        // Apply the new joint configuration to the rod segment terminals.
        m_linesearch_base.  setDoFs(  m_linesearch_base.getDoFs(), true /* set spatially coherent thetas */);
        m_base.             setDoFs(             m_base.getDoFs(), true /* set spatially coherent thetas */);
        // m_diff_linkage_flat.setDoFs(m_diff_linkage_flat.getDoFs(), true /* set spatially coherent thetas */);

        m_linesearch_deployed.  set_interleaving_type(new_type);
        m_deployed.             set_interleaving_type(new_type);
        m_diff_linkage_deployed.set_interleaving_type(new_type);

        // Apply the new joint configuration to the rod segment terminals.
        m_linesearch_deployed.  setDoFs(  m_linesearch_deployed.getDoFs(), true /* set spatially coherent thetas */);
        m_deployed.             setDoFs(             m_deployed.getDoFs(), true /* set spatially coherent thetas */);
        m_diff_linkage_deployed.setDoFs(m_diff_linkage_deployed.getDoFs(), true /* set spatially coherent thetas */);
    }

    void setCShellOptimization(Object<Real> &flat, Object<Real> &deployed) {
        m_numParams = flat.numDesignParams();
        m_E0 = deployed.energy();  
        m_linesearch_base.set(flat);
        m_linesearch_deployed.set(deployed);

        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }
    // Change the deployed linkage's opening angle by "alpha", resolving for equilibrium.
    // Side effect: commits the linesearch linkage (like calling newPt)
    void setTargetAngle(Real alpha) {
        m_alpha_tgt = alpha;
        m_deployed_optimizer->get_problem().setLEQConstraintRHS(alpha);

        m_linesearch_base    .set(m_base    );
        m_linesearch_deployed.set(m_deployed);

        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void setAllowFlatActuation(bool allow) {
        m_allowFlatActuation = allow;
        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void commitLinesearchLinkage() {
        m_base    .set(m_linesearch_base);
        m_deployed.set(m_linesearch_deployed);
        // Stash the current factorizations to be reused at each step of the linesearch
        // to predict the equilibrium at the new design parameters.
        getDeployedOptimizer().solver.stashFactorization();
    }

    Real getTargetAngle() const { return m_alpha_tgt; }


    // Construct/update a target surface for the surface fitting term by
    // inferring a smooth surface from the current joint positions.
    // Also update the closest point projections.
    void constructTargetSurface(size_t loop_subdivisions = 0, Eigen::Vector3d scale_factors = Eigen::Vector3d(1, 1, 1)) {
        target_surface_fitter.constructTargetSurface(m_linesearch_deployed, loop_subdivisions, scale_factors);
        invalidateAdjointState();
    }

    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) {
        getDeployedOptimizer().options = eopts;
    }

    NewtonOptimizerOptions getEquilibriumOptions() const {
        return m_deployed_optimizer->options;
    }

    NewtonOptimizer &getDeployedOptimizer() { return *m_deployed_optimizer; }

    const LOMinAngleConstraint<Object> &getMinAngleConstraint() const {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    LOMinAngleConstraint<Object> &getMinAngleConstraint() {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }
    // Write the full, dense Hessians of J and angle_constraint to a file.
    void dumpHessians(const std::string &hess_J_path, const std::string &hess_ac_path, Real fd_eps = 1e-5);


    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    Real gamma = 0.9;
    void setGamma(Real val)                        { gamma = val; objective.get("ElasticEnergyFlat").setWeight(val / m_E0); invalidateAdjointState();}
    Real getGamma()                        const { return objective.get("ElasticEnergyFlat").getWeight() * m_E0; }


private:
    void m_forceEquilibriumUpdate();
    // Return whether "params" are actually new...
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params);

    // Update the closest point projections for each joint to the target surface.
    void m_updateClosestPoints() { target_surface_fitter.updateClosestPoints(m_linesearch_deployed); }

    // Check if the minimum angle constraint is active and if so, change the closed
    // configuration's actuation angle to satisfy the constraint.
    // void m_updateMinAngleConstraintActuation();

    // Update the adjoint state vectors "w"
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params);

    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXd m_w_x; // adjoint state vectors
    Eigen::VectorXd m_delta_w_x, m_delta_x3d; // variations of adjoint/forward state from the last call to apply_hess (for debugging)
    Eigen::VectorXd m_delta_delta_x3d;   // second variations of forward state from last call to m_updateEquilibrium (for debugging)
    Eigen::VectorXd m_second_order_x3d; // second-order predictions of the linkage's equilibrium (for debugging)
    Eigen::VectorXd m_d3E_w;
    Eigen::VectorXd m_w_rhs, m_delta_w_rhs;
    Real m_alpha_tgt = 0.0;
    Object<Real> &m_deployed; // m_flat is defined in the base class as m_base.
    Object<Real> m_linesearch_deployed; // m_linesearch_flat is defined in the base class as m_linesearch_base.
    std::unique_ptr<NewtonOptimizer> m_deployed_optimizer;

    std::unique_ptr<LOMinAngleConstraint<Object>> m_minAngleConstraint;

    Object<ADReal> m_diff_linkage_deployed;

    bool m_allowFlatActuation = true; // whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.

};

#include "CShellOptimization.inl"
#endif /* end of include guard: CSHELLOPTIMIZATION_HH */
