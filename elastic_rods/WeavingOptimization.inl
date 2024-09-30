#include "WeavingOptimization.hh"

template<template<typename> class Object>
WeavingOptimization<Object>::WeavingOptimization(Object<Real> &weaver, const std::string input_surface_path, bool useCenterline, const NewtonOptimizerOptions &eopts, int pinJoint, bool useFixedJoint, const std::vector<size_t> &fixedVars)
    : LinkageOptimization<Object>(weaver, eopts, weaver.energy(), BBox<Point3D>(weaver.deformedPoints()).dimensions().norm(), weaver.totalRestLength()),
      m_restKappaSmoothing(m_linesearch_base), m_restLengthMinimization(m_linesearch_base)
{
    std::runtime_error mismatch("Linkage mismatch");
    if (m_numParams != weaver.numDesignParams()) throw mismatch;
    // Initialize the auto diff
    m_diff_linkage_weaver.set(weaver);

    // Create the objective terms
    using OET = OptEnergyType;
    using EEO = ElasticEnergyObjective<Object>;
    using TSF = TargetFittingDOOT<Object>;
    using RLM = RegularizationTermDOOWrapper<Object, RestLengthMinimization>;
    using RCS = RegularizationTermDOOWrapper<Object, RestCurvatureSmoothing>;
    using CFO = ContactForceObjective<Object>;
    auto &tsf = target_surface_fitter;
    objective.add("ElasticEnergy",          OET::ElasticBase,    std::make_shared<EEO>(m_linesearch_base), 1.0 / m_E0);
    objective.add("TargetFitting",          OET::Target,         std::make_shared<TSF>(m_linesearch_base, tsf), beta / (m_l0 * m_l0));
    objective.add("RestLengthMinimization", OET::Regularization, std::make_shared<RLM>(m_linesearch_base),  1.0);
    objective.add("RestCurvatureSmoothing", OET::Smoothing,      std::make_shared<RCS>(m_linesearch_base), 10.0);
    objective.add("ContactForce",           OET::ContactForce,   std::make_shared<CFO>(m_linesearch_base),  0.0);

    // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
    target_surface_fitter.joint_pos_tgt = weaver.jointPositions();
    loadTargetSurface(input_surface_path);

    m_rigidMotionFixedVars.insert(std::end(m_rigidMotionFixedVars), std::begin(fixedVars), std::end(fixedVars));

    if (useFixedJoint) {
        // Constrain the position and orientation of the centermost joint to prevent global rigid motion.
        if (pinJoint != -1) {
            m_rm_constrained_joint = pinJoint;
            if (m_rm_constrained_joint >= weaver.numJoints()) throw std::runtime_error("Manually specified pinJoint is out of bounds");
        }
        else {
            m_rm_constrained_joint = weaver.centralJoint();
        }
        const size_t jdo = weaver.dofOffsetForJoint(m_rm_constrained_joint);
        for (size_t i = 0; i < 6; ++i) m_rigidMotionFixedVars.push_back(jdo + i);
    }
    m_weaver_optimizer = get_equilibrium_optimizer(m_linesearch_base, TARGET_ANGLE_NONE, m_rigidMotionFixedVars);

    m_weaver_optimizer->options = m_equilibrium_options;

    // Trade off between fitting to the individual joint targets and the target surface.
    target_surface_fitter.setUseCenterline(weaver, useCenterline, 0.01);

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();

    commitLinesearchLinkage();
}

template<template<typename> class Object>
WeavingOptimization<Object>::WeavingOptimization(Object<Real> &weaver, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, bool useCenterline, const NewtonOptimizerOptions &eopts, int pinJoint, bool useFixedJoint, const std::vector<size_t> &fixedVars)
    : LinkageOptimization<Object>(weaver, eopts, weaver.energy(), BBox<Point3D>(weaver.deformedPoints()).dimensions().norm(), weaver.totalRestLength()),
      m_restKappaSmoothing(m_linesearch_base), m_restLengthMinimization(m_linesearch_base)
{
    std::runtime_error mismatch("Linkage mismatch");
    if (m_numParams != weaver.numDesignParams()) throw mismatch;
    // Initialize the auto diff
    m_diff_linkage_weaver.set(weaver);

    // Create the objective terms
    using OET = OptEnergyType;
    using EEO = ElasticEnergyObjective<Object>;
    using TSF = TargetFittingDOOT<Object>;
    using RLM = RegularizationTermDOOWrapper<Object, RestLengthMinimization>;
    using RCS = RegularizationTermDOOWrapper<Object, RestCurvatureSmoothing>;
    using CFO = ContactForceObjective<Object>;
    auto &tsf = target_surface_fitter;
    objective.add("ElasticEnergy",          OET::ElasticBase,    std::make_shared<EEO>(m_linesearch_base), 1.0 / m_E0);
    objective.add("TargetFitting",          OET::Target,         std::make_shared<TSF>(m_linesearch_base, tsf), beta / (m_l0 * m_l0));
    objective.add("RestLengthMinimization", OET::Regularization, std::make_shared<RLM>(m_linesearch_base),  1.0);
    objective.add("RestCurvatureSmoothing", OET::Smoothing,      std::make_shared<RCS>(m_linesearch_base), 10.0);
    objective.add("ContactForce",           OET::ContactForce,   std::make_shared<CFO>(m_linesearch_base),  0.0);

    // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
    target_surface_fitter.joint_pos_tgt = weaver.jointPositions();
    loadTargetSurfaceFromData(V,F);

    m_rigidMotionFixedVars.insert(std::end(m_rigidMotionFixedVars), std::begin(fixedVars), std::end(fixedVars));

    if (useFixedJoint) {
        // Constrain the position and orientation of the centermost joint to prevent global rigid motion.
        if (pinJoint != -1) {
            m_rm_constrained_joint = pinJoint;
            if (m_rm_constrained_joint >= weaver.numJoints()) throw std::runtime_error("Manually specified pinJoint is out of bounds");
        }
        else {
            m_rm_constrained_joint = weaver.centralJoint();
        }
        const size_t jdo = weaver.dofOffsetForJoint(m_rm_constrained_joint);
        for (size_t i = 0; i < 6; ++i) m_rigidMotionFixedVars.push_back(jdo + i);
    }
    m_weaver_optimizer = get_equilibrium_optimizer(m_linesearch_base, TARGET_ANGLE_NONE, m_rigidMotionFixedVars);

    m_weaver_optimizer->options = m_equilibrium_options;

    // Trade off between fitting to the individual joint targets and the target surface.
    target_surface_fitter.setUseCenterline(weaver, useCenterline, 0.01);

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();

    commitLinesearchLinkage();
}

template<template<typename> class Object>
void WeavingOptimization<Object>::m_forceEquilibriumUpdate() {
    // Update the weaver linkage equilibria
    m_equilibriumSolveSuccessful = true;
    try {
        if (m_equilibrium_options.verbose)
            std::cout << "Weaver equilibrium solve" << std::endl;
        auto cr = getWeaverOptimizer().optimize();
        // A backtracking failure will happen if the gradient tolerance is set too low
        // and generally does not indicate a complete failure/bad estimate of the equilibrium.
        // We therefore accept such equilibria with a warning.
        // (We would prefer to reject saddle points, but the surface-attracted structures
        //  appear to //  sometimes have backtracking failures in saddle points
        //  close to reasonably stable equilibria...)
        bool acceptable_failed_equilibrium = cr.backtracking_failure; //  && !cr.indefinite.back();
        // std::cout << "cr.backtracking_failure: " << cr.backtracking_failure << std::endl;
        // std::cout << "cr.indefinite.back(): " << cr.indefinite.back() << std::endl;
        if (!cr.success && !acceptable_failed_equilibrium) {
            throw std::runtime_error("Equilibrium solve did not converge");
        }
        if (acceptable_failed_equilibrium) {
            std::cout << "WARNING: equillibrium solve backtracking failure." << std::endl;
        }
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return; // subsequent update_factorizations will fail if we caught a Tau runaway...
    }

    // We will be evaluating the Hessian/using the simplified gradient expressions:
    m_linesearch_base.updateSourceFrame();
    m_linesearch_base.updateRotationParametrizations();
    // Use the final equilibria's Hessians for sensitivity analysis, not the second-to-last iterates'
    try {
        getWeaverOptimizer().update_factorizations();
    }
    catch (const std::runtime_error &e) {
        std::cout << "Hessian factorization at equilibrium failed failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return;
    }

    // The cached adjoint state is invalidated whenever the equilibrium is updated...
    m_adjointStateIsCurrent      = false;
    m_autodiffLinkagesAreCurrent = false;

    objective.update();
}

template<template<typename> class Object>
bool WeavingOptimization<Object>::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams) {
    if ((m_linesearch_base.getDesignParameters() - newParams).norm() < 1e-16) return false;
    m_linesearch_base.set(m_base);

    const Eigen::VectorXd currParams = m_base.getDesignParameters();
    Eigen::VectorXd delta_p = newParams - currParams;

    if (delta_p.norm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        m_forceEquilibriumUpdate();
        return true;
        // The following caching is not working for some reason...
        // TODO: debug
        // auto &opt_2D = getFlatOptimizer();
        // auto &opt_3D = getDeployedOptimizer();

        // if (!(opt_2D.solver.hasStashedFactorization() && opt_3D.solver.hasStashedFactorization()))
        //     throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");

        // // Copy the stashed factorization into the current one (preserving the stash)
        // opt_2D.solver.swapStashedFactorization();
        // opt_3D.solver.swapStashedFactorization();
        // opt_2D.solver.stashFactorization();
        // opt_3D.solver.stashFactorization();

        // // The cached adjoint state is invalidated whenever the equilibrium is updated...
        // m_adjointStateIsCurrent      = false;
        // m_autodiffLinkagesAreCurrent = false;

        // m_updateClosestPoints();
        //
        // return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the current equilibrium).
    // We will only replace this equilibrium if the higher order predictions achieve a lower energy.
    m_linesearch_base.setDesignParameters(newParams);
    Real bestEnergy3d = m_linesearch_base.energy();
    Eigen::VectorXd curr_x = m_base.getDoFs();
    Eigen::VectorXd best_x = curr_x;
    if (prediction_order > PredictionOrder::Zero) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_weaver = getWeaverOptimizer();
        if (!opt_weaver.solver.hasStashedFactorization())
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
        opt_weaver.solver.swapStashedFactorization();

        {
            // Solve for equilibrium perturbation corresponding to delta_p:
            //      [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
            //      [a^T  0][delta lambda]   [        0        ]
            //                               \_________________/
            //                                        b
            const size_t np = numParams(), nd = m_base.numDoF();
            VecX_T<Real> neg_deltap_padded(nd + np);
            neg_deltap_padded.setZero();
            neg_deltap_padded.tail(np) = -delta_p;

            // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
            HessianComputationMask mask_dxdp;
            mask_dxdp.dof_in      = false;
            mask_dxdp.designParameter_out = false;

            m_delta_x = opt_weaver.extractFullSolution(opt_weaver.solver.solve(opt_weaver.removeFixedEntries(m_base.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd))));

            // Evaluate the energy at the 1st order-predicted equilibrium
            {
                auto first_order_x3d = (curr_x + m_delta_x).eval();
                     // first_order_x2d = (curr_x2d + m_delta_x2d).eval();
                m_linesearch_base.setDoFs(first_order_x3d);
                Real energy1stOrder3d = m_linesearch_base.energy();
                if (energy1stOrder3d < bestEnergy3d) { std::cout << " used first order prediction, energy reduction " << bestEnergy3d - energy1stOrder3d << std::endl; bestEnergy3d = energy1stOrder3d; best_x = first_order_x3d; } else { m_linesearch_base.setDoFs(best_x); }
            }

            if (prediction_order > PredictionOrder::One) {
                // TODO: also stash autodiff linkages for committed linkages?
                // Solve for perturbation of equilibrium perturbation corresponding to delta_p:
                //      [H_3D a][delta_p^T d2x/dp^2 delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p 0][delta_x     ] + [-(d3E/dxdpdx delta_x + d3E/dxdpdp delta_p) delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p    d3E/dxdpdx delta_x + d3E/dxdpdp delta_p    0][delta_x     ]
                //      [a^T  0][delta_p^T d2l/dp^2 delta_p]    [0                                   0][delta_lambda] + [                       0                          ]    [                0                                         0                       0][delta_p     ]
                //                                                                                                                                                                                                                                                   [delta_lambda]
                m_diff_linkage_weaver.set(m_base);

                Eigen::VectorXd neg_d3E_delta_x;
                {
                    // inject design parameter perturbation.
                    VecX_T<ADReal> ad_p = currParams;
                    for (size_t i = 0; i < np; ++i) ad_p[i].derivatives()[0] = delta_p[i];
                    m_diff_linkage_weaver.setDesignParameters(ad_p);

                    // inject equilibrium perturbation
                    VecX_T<ADReal> ad_x = curr_x;
                    for (int i = 0; i < ad_x.size(); ++i) ad_x[i].derivatives()[0] = m_delta_x[i];
                    m_diff_linkage_weaver.setDoFs(ad_x);

                    VecX_T<Real> delta_edof_3d(nd + np);
                    delta_edof_3d.head(nd) = m_delta_x;
                    delta_edof_3d.tail(np) = delta_p;

                    neg_d3E_delta_x = -extractDirectionalDerivative(m_diff_linkage_weaver.applyHessianPerSegmentRestlen(delta_edof_3d)).head(nd);
                }

                m_delta_delta_x = opt_weaver.extractFullSolution(opt_weaver.solver.solve(opt_weaver.removeFixedEntries(neg_d3E_delta_x)));

                // Evaluate the energy at the 2nd order-predicted equilibrium, roll back to previous best if energy is higher.
                {
                    m_second_order_x = (curr_x + m_delta_x + 0.5 * m_delta_delta_x).eval(),
                    m_linesearch_base.setDoFs(m_second_order_x);
                    Real energy2ndOrder3d = m_linesearch_base.energy();
                    if (energy2ndOrder3d < bestEnergy3d) { std::cout << " used second order prediction, energy reduction " << bestEnergy3d - energy2ndOrder3d << std::endl; bestEnergy3d = energy2ndOrder3d; best_x = m_second_order_x;} else { m_linesearch_base.setDoFs(best_x); }
                }
            }
        }

        // Return to using the primary factorization, storing the committed
        // linkages' factorizations back in the stash for later use.
        opt_weaver.solver.swapStashedFactorization();
    }

    m_forceEquilibriumUpdate();

    return true;
}

// Update the adjoint state vectors "w", "y", and "s"
template<template<typename> class Object>
bool WeavingOptimization<Object>::m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params) {
    m_updateEquilibria(params);
    if (m_adjointStateIsCurrent) return false;
    std::cout << "Updating adjoint state" << std::endl;

    // Solve the adjoint problems needed to efficiently evaluate the gradient.
    // Note: if the Hessian modification failed (tau runaway), the adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the adjoint state to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    try {
        // Adjoint solve for the target fitting objective on the deployed linkage
        objective.updateAdjointState(getWeaverOptimizer());
    }
    catch (...) {
        std::cout << "WARNING: Adjoint state solve failed" << std::endl;
        objective.clearAdjointState();
    }

    m_adjointStateIsCurrent = true;

    return true;
}

template<template<typename> class Object>
Eigen::VectorXd WeavingOptimization<Object>::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    m_updateAdjointState(params);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    const size_t nd = m_linesearch_base.numDoF();
    const size_t np = numParams();
    Eigen::VectorXd w_padded(nd + np);
    w_padded.head(nd) = objective.adjointState();
    w_padded.tail(np).setZero();
    return objective.grad_p(opt_eType) - m_linesearch_base.applyHessianPerSegmentRestlen(w_padded, mask).tail(numParams());
}

template<template<typename> class Object>
Eigen::VectorXd WeavingOptimization<Object>::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params,
                                                const Eigen::Ref<const Eigen::VectorXd> &delta_p,
                                                Real coeff_J, Real /* coeff_c */, Real /* coeff_angle_constraint */, OptEnergyType opt_eType) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess_J");
    BENCHMARK_START_TIMER_SECTION("Preamble");
    const size_t np = numParams(), nd = m_linesearch_base.numDoF();
    if (size_t( params.size()) != np) throw std::runtime_error("Incorrect parameter vector size");
    if (size_t(delta_p.size()) != np) throw std::runtime_error("Incorrect delta parameter vector size");
    m_updateAdjointState(params);

    if (!m_autodiffLinkagesAreCurrent) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Update autodiff linkages");
        m_diff_linkage_weaver.set(m_linesearch_base);
        m_autodiffLinkagesAreCurrent = true;
    }

    auto &opt  = getWeaverOptimizer();
    auto &H_3D = opt.solver;

    BENCHMARK_STOP_TIMER_SECTION("Preamble");

    VecX_T<Real> neg_deltap_padded(nd + np);
    neg_deltap_padded.head(nd).setZero();
    neg_deltap_padded.tail(np) = -delta_p;

    // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
    HessianComputationMask mask_dxdp, mask_dxpdx;
    mask_dxdp.dof_in      = false;
    mask_dxdp.designParameter_out = false;
    mask_dxpdx.designParameter_in = false;

    VecX_T<Real> delta_dJ_dxp;

    try {
        // Solve for state perturbation
        // [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        // [a^T  0][delta lambda]   [        0        ]
        //                          \_________________/
        //                                   b
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x");
            VecX_T<Real> b = m_linesearch_base.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd);
            m_delta_x = opt.extractFullSolution(H_3D.solve(opt.removeFixedEntries(b)));
        }

        VecX_T<Real> delta_xp(nd + np);
        delta_xp.head(nd) = m_delta_x;
        delta_xp.tail(np) = delta_p;

        // Solve for adjoint state perturbation
        BENCHMARK_START_TIMER_SECTION("getDoFs and inject state");
        VecX_T<ADReal> ad_xp = m_linesearch_base.getExtendedDoFsPSRL();
        for (size_t i = 0; i < np + nd; ++i) ad_xp[i].derivatives()[0] = delta_xp[i];
        m_diff_linkage_weaver.setExtendedDoFsPSRL(ad_xp);
        BENCHMARK_STOP_TIMER_SECTION("getDoFs and inject state");

        BENCHMARK_START_TIMER_SECTION("delta_dJ_dxp");
        delta_dJ_dxp = objective.delta_grad(delta_xp, m_diff_linkage_weaver, opt_eType);
        BENCHMARK_STOP_TIMER_SECTION("delta_dJ_dxp");

        // Solve for adjoint state perturbation
        // [H_3D a][delta w_x     ] = [ d^2J/dpdxp delta_xp ] - [d3E/dx dx dxp delta_xp] w
        // [a^T  0][delta w_lambda]   [           0         ]   [           0          ]
        //                            \__________________________________________________/
        //                                                        b
        if (coeff_J != 0.0) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta w x");
            BENCHMARK_START_TIMER_SECTION("Hw");
            VecX_T<ADReal> w_padded(nd + np);
            w_padded.head(nd) = objective.adjointState();
            w_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_w for evaluating the full Hessian matvec expressions below...
            m_d3E_w = extractDirectionalDerivative(m_diff_linkage_weaver.applyHessianPerSegmentRestlen(w_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hw");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");

            auto b = (delta_dJ_dxp.head(nd) - m_d3E_w.head(nd)).eval();

            if (opt.get_problem().hasLEQConstraint()) m_delta_w_x = opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(b)));
            else                                      m_delta_w_x = opt.extractFullSolution(          opt.solver.solve(opt.removeFixedEntries(b)));

            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }
    }
    catch (...) {
        m_delta_x   = VecX_T<Real>::Zero(nd     );
        m_delta_w_x = VecX_T<Real>::Zero(nd     );
        m_d3E_w     = VecX_T<Real>::Zero(nd + np);
    }

    VecX_T<Real> result;
    result.setZero(np);
    // Accumulate the J hessian matvec
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer3("evaluate hessian matvec");
        if (coeff_J != 0.0) {
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = m_delta_w_x;
            delta_edofs.tail(np).setZero();

            HessianComputationMask mask;
            mask.dof_out = false;
            mask.designParameter_in = false;

            result += delta_dJ_dxp.tail(np)
                   -  m_linesearch_base.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np)
                   -  m_d3E_w.tail(np);
            result *= coeff_J;
        }
    }

    return result;
}
