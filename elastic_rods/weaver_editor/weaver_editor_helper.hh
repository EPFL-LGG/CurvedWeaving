#ifndef WEAVEREDITORHELPER_HH
#define WEAVEREDITORHELPER_HH

#include <igl/opengl/glfw/Viewer.h>
#include <GLFW/glfw3.h>
#include "../SurfaceAttractedLinkage.hh"
#include "../WeavingOptimization.hh"
#include "../design_parameter_solve.hh"
#include "../infer_target_surface.hh"
#include "../open_linkage.hh"
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/unused.hh>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/file_dialog_open.h>
#include <imgui/imgui.h>


#include <string>
#include <iostream>
#include <map>
#include <thread>

using Viewer     = igl::opengl::glfw::Viewer;
using ViewerCore = igl::opengl::ViewerCore;
   
using Linkage_dPC = DesignParameterConfig;


// Global flag used to interrupt optimization; setting this to true will
// cause Knitro to quit when the current iteration completes.
bool optimization_cancelled = false;
bool optimization_running = false;
bool needs_redraw = false;
bool requestHessianDump = false;

size_t numHessDump = 0;

double fd_eps = 1e-3;

template<typename Object>
void getLinkageMesh(const Object &l, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    static std::vector<MeshIO::IOVertex > vertices;
    static std::vector<MeshIO::IOElement> elements;
    vertices.clear();
    elements.clear();
    l.visualizationGeometry(vertices, elements, true);
    meshio_to_igl(vertices, elements, V, F);
    // std::cout << "getLinkageMesh" << std::endl;
    // static size_t i = 0;
    // l.saveVisualizationGeometry("visMesh_" + std::to_string(i++) + ".msh");
}

template<template<typename> class Object>
void dumpHessians(WeavingOptimization<Object> &lopt) {
    lopt.dumpHessians("hess_J_"  + std::to_string(numHessDump) + ".txt",
                      "hess_ac_" + std::to_string(numHessDump) + ".txt",
                      fd_eps);
    ++numHessDump;
}

template<template<typename> class Object>
struct CachedStats {
    Real J, J_target, gradp_J_norm, J_regularization, J_smoothing, E_weaver, E_flat, E_weaver_rod_max,
         flatness;

    void update(WeavingOptimization<Object> &lopt) {
        J                   = lopt.J();
        J_target            = lopt.LinkageOptimization<Object>::J_target();
        gradp_J_norm        = lopt.LinkageOptimization<Object>::gradp_J().norm();
        J_regularization    = lopt.J_regularization();
        J_smoothing         = lopt.J_smoothing();
        E_weaver          = lopt.getLinesearchWeaverLinkage().energy();
        E_weaver_rod_max  = lopt.getLinesearchWeaverLinkage().maxRodEnergy();
    }
};

CachedStats<SurfaceAttractedLinkage_T> stats;

template<template<typename> class Object>
struct WOptKnitroNewPtCallback : public NewPtCallbackBase {
    WOptKnitroNewPtCallback(WeavingOptimization<Object> &lopt) : m_lopt(lopt) { }

    virtual int operator()(const double *x) override {
        const size_t np = m_lopt.numParams();
        m_lopt.newPt(Eigen::Map<const Eigen::VectorXd>(x, np));
        if (requestHessianDump) { dumpHessians(m_lopt); }
        needs_redraw = true;
        stats.update(m_lopt);
        glfwPostEmptyEvent(); // Run another iteration of the event loop so the viewer redraws.
        return (optimization_cancelled) ? KPREFIX(RC_USER_TERMINATION) : 0;
    }
private:
    std::function<void()> m_update_viewer;
    WeavingOptimization<Object> &m_lopt;
};

template<template<typename> class Object>
void optimize(OptAlgorithm alg, WeavingOptimization<Object> &lopt, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, double minRestLen) {
    OptKnitroProblem<Object> problem(lopt, false, false, minRestLen);

    std::vector<Real> x_init(lopt.numParams());
    Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size()) = lopt.getLinesearchWeaverLinkage().getDesignParameters();
    problem.setXInitial(x_init);

    optimization_cancelled = false;
    optimization_running = true;
    WOptKnitroNewPtCallback<Object> callback(lopt);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    // exact first and second derivatives; no KPREFIX(GRADOPT_*) or KPREFIX(HESSOPT_*) parameter is needed.
    int hessopt = 0;
    if (alg == OptAlgorithm::NEWTON_CG) hessopt = 5;  // exact Hessian-vector products
    else if (alg == OptAlgorithm::BFGS) hessopt = 2;  // BFGS approximation
    else throw std::runtime_error("Unknown algorithm");

    KnitroSolver solver(&problem, /* exact gradients */ 1, hessopt);
    solver.useNewptCallback();
    solver.setParam(KPREFIX(PARAM_HONORBNDS), KPREFIX(HONORBNDS_ALWAYS)); // always respect bounds during optimization
    solver.setParam(KPREFIX(PARAM_MAXIT), int(num_steps));
    solver.setParam(KPREFIX(PARAM_PRESOLVE), KPREFIX(PRESOLVE_NONE));
    solver.setParam(KPREFIX(PARAM_DELTA), trust_region_scale);
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK), KPREFIX(DERIVCHECK_ALL));
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK_TYPE), KPREFIX(DERIVCHECK_CENTRAL));
    // solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_BAR_DIRECT));   // interior point with exact Hessian
    solver.setParam(KPREFIX(PARAM_PAR_NUMTHREADS), 12);
    solver.setParam(KPREFIX(PARAM_HESSIAN_NO_F), KPREFIX(HESSIAN_NO_F_ALLOW)); // allow Knitro to call our hessvec with sigma = 0
    // solver.setParam(KPREFIX(PARAM_LINSOLVER), KPREFIX(LINSOLVER_MKLPARDISO));
    solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_ACT_CG));
    solver.setParam(KPREFIX(PARAM_ACT_QPALG), KPREFIX(ACT_QPALG_ACT_CG)); // default ended up choosing KPREFIX(ACT_QPALG_BAR_DIRECT)
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), 25);
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), int(lopt.numParams())); // TODO: decide on this.
    // solver.setParam(KPREFIX(PARAM_BAR_FEASIBLE), KPREFIX(BAR_FEASIBLE_NO));

    solver.setParam(KPREFIX(PARAM_OPTTOL), optimality_tol);
    solver.setParam(KPREFIX(PARAM_OUTLEV), KPREFIX(OUTLEV_ALL));

    try {
        BENCHMARK_RESET();
        int solveStatus = solver.solve();
        BENCHMARK_REPORT_NO_MESSAGES();

        if (solveStatus != 0) {
            std::cout << std::endl;
            std::cout << "KNITRO failed to solve the problem, final status = ";
            std::cout << solveStatus << std::endl;
        }
    }
    catch (KnitroException &e) {
        problem.setNewPointCallback(nullptr);
        printKnitroException(e);
        optimization_running = false;
        throw e;
    }
    problem.setNewPointCallback(nullptr);

    optimization_running = false;
}
#endif /* end of include guard: WEAVEREDITORHELPER_HH */
