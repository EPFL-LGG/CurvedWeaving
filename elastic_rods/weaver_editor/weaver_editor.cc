#include "weaver_editor_helper.hh"
#include "weaver_editor_set_up.hh"

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


int main(int argc, char * argv[])
{
    if ((argc != 4) && (argc != 5)) {
        std::cerr << "Usage: linkage_editor linkage.obj cross_section.json input_surface obj" << std::endl;
        exit(-1);
    }

    const size_t width = 1280;
    const size_t height = 800;

    // Step size for gradient descent
    size_t num_steps = 2000;
    OptAlgorithm opt_algorithm = OptAlgorithm::NEWTON_CG;
    double trust_region_scale = 1.0;
    PredictionOrder prediction_order = PredictionOrder::Two;

    double jointPosWeight = 0.01,              // controls trade-off between fitting to target joint positions and fitting to target surface
           jointPosValence2Multiplier = 10.0; // controls whether we try harder to fit valence 2 vertices to their target positions
    bool useCenterline = true;
    bool showTargetSurface = false;

    // bool setting_deployment_angle = false;
    // double target_deployment_angle = 0.0;

    double designOptimizationTol = 1e-2;

    double linkage_attraction_weight = 100;

    bool useRestKappa = true;
    bool useFixedJoint = false;
    bool holdClosestPointsFixed = true;
    bool useWeaving = true;
    bool useXshell = false;
    bool noOffset = false;
    // Construct the flat and deployed linkage
    const std::string linkage_path(argv[1]);
    const std::string cross_section_path(argv[2]);
    const std::string input_surface_path(argv[3]);

    // std::vector<MeshIO::IOVertex > vertices;
    // std::vector<MeshIO::IOElement> edges;
    // MeshIO::load(linkage_path, vertices, edges);

    // Define surface attracted linkage with the input surface and linkage graph.
    SurfaceAttractedLinkage weaverLinkage(input_surface_path, true, linkage_path, 20, false, InterleavingType::weaving);
    auto save_tgt_joint_pos = weaverLinkage.jointPositions();
    setUpLinkageForOptimization(weaverLinkage, linkage_attraction_weight, holdClosestPointsFixed, cross_section_path, useFixedJoint);

    NewtonOptimizerOptions eopts;
    eopts.gradTol = 1e-6;
    eopts.verbose = 10;
    eopts.beta = 1e-8;
    eopts.niter = 100;
    WeavingOptimization<SurfaceAttractedLinkage_T> lopt(weaverLinkage, input_surface_path, useCenterline, eopts, 0, useFixedJoint);

    lopt.target_surface_fitter.joint_pos_tgt = save_tgt_joint_pos;

    BENCHMARK_REPORT_NO_MESSAGES();

    stats.update(lopt);

    // Create viewer with two empty mesh slots.
    Viewer viewer;
    viewer.append_mesh();
    const int weaver_mesh_id = 1, target_surface_mesh_id = 0;

    // Scratch space for updateLinkageMeshes
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    auto updateLinkageMeshes = [&](){
        getLinkageMesh(weaverLinkage, V, F);
        viewer.data(weaver_mesh_id).set_mesh(V, F);
        viewer.data(weaver_mesh_id).face_based = true;
        viewer.data(weaver_mesh_id).show_lines = false;
        viewer.data(weaver_mesh_id).set_colors(Eigen::RowVector3d(0.65, 0.65, 0.65));
    };

    auto updateTargetMesh = [&]() {
        viewer.data(target_surface_mesh_id).clear();
        viewer.data(target_surface_mesh_id).set_mesh(lopt.target_surface_fitter.getV(), lopt.target_surface_fitter.getF());
        viewer.data(target_surface_mesh_id).face_based = true;
        viewer.data(target_surface_mesh_id).show_lines = false;
        viewer.data(target_surface_mesh_id).set_colors(Eigen::RowVector3d(0.95, 0.95, 0.95));
    };

    updateLinkageMeshes();
    updateTargetMesh();

    int view;
    viewer.callback_init = [&](Viewer &)
    {
        glfwSetWindowTitle(viewer.window, "Weaving Editor");
        view = viewer.core_list[0].id; // will be resized by callback_post_resize
        viewer.core(view).background_color << 0.92, 0.92, 0.92, 1.0;

        viewer.core(view).rotation_type = ViewerCore::ROTATION_TYPE_TRACKBALL;

        viewer.core(view).camera_dnear = 0.005;
        viewer.core(view).camera_dfar  = 50;
        // Show weaving linkage
        viewer.data(weaver_mesh_id).set_visible(true, view);

        // Initially the target surface mesh is invisible (in both views)
        // viewer.data(target_surface_mesh_id).set_visible(false, top_view);
        viewer.data(target_surface_mesh_id).set_visible(false, view);

        return false; // also init the plugins
    };

    // Update meshes before redrawing if the optimizer has updated the linkage
    // state.
    viewer.callback_pre_draw = [&](Viewer &/* v */) {
        if (needs_redraw) {
            updateLinkageMeshes();
            needs_redraw = false;
        }
        return false;
    };

    std::thread optimization_thread;

    viewer.callback_key_pressed = [&](Viewer &, unsigned int key, int /* mod */)
    {

        if ((key == 'g') || (key == 'G')) {
            if (!optimization_running) {
                optimization_running = true;
                if (optimization_thread.joinable())
                    optimization_thread.join(); // shouldn't actually happen...
                optimization_thread = std::thread(optimize<SurfaceAttractedLinkage_T>, opt_algorithm, std::ref(lopt), num_steps, trust_region_scale, designOptimizationTol, -1);
            }
            return true;
        }

        if ((key == 'h') || (key == 'H')) {
            if (!optimization_running) dumpHessians(lopt);
            else                       requestHessianDump = true;
            return true;
        }

        if ((key == 'c') || (key == 'C')) {
            optimization_cancelled = true;
            return true;
        }

        // if ((key == 'f') || (key == 'f')) {
        //  auto params = lopt.getLinesearchFlatLinkage().getDesignParameters();
        //  auto grad = lopt.gradp_J(params);
        //     Real eps = 1e-4;
        //     params[0] += eps;
        //     Real Jplus = lopt.J(params);
        //     params[0] -= 2 * eps;
        //     Real Jminus = lopt.J(params);
        //     std::cout.precision(19);
        //     std::cout << "fd diff J" << (Jplus - Jminus) / (2 * eps) << std::endl;
        //     std::cout << "grad    J" << grad[0] << std::endl;

        //     return true;
        // }



        return false;
    };

    viewer.callback_post_resize = [&](Viewer &v, int w, int h) {
        v.core(view).viewport = Eigen::Vector4f(0, 0, w, h);
        return true;
    };

    // Initialize the split views' viewports
    viewer.callback_post_resize(viewer, width, height);

    ////////////////////////////////////////////////////////////////////////////
    // IMGui UI
    ////////////////////////////////////////////////////////////////////////////
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_window = [&]() { };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]()
    {
        std::string title = optimization_running ? "Optimizer - Running" : "Optimizer";
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(10, 10),    ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 600), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Optimizer", nullptr,         ImGuiWindowFlags_NoSavedSettings);
        if (optimization_running) ImGui::StyleColorsLight();
        else                      ImGui::StyleColorsDark();

        ImGui::PushItemWidth(-80);

        if (optimization_running) {
            if (ImGui::Button("Request Stop", ImVec2(-1,0)))
                optimization_cancelled = true;
        }
        else {
            // Optimization settings are disabled when the optimization is running.
            bool updated = false;

            const double beta_min = 0;
            static double beta = lopt.getBeta();
            if (ImGui::DragScalar("beta",  ImGuiDataType_Double, &beta,  10, &beta_min, 0, "%.4f")) {
                lopt.setBeta(beta);
                updated = true;
            }

            const double gamma_min = 0, gamma_max = 100;
            static double gamma = lopt.getBeta();
            if (ImGui::DragScalar("gamma", ImGuiDataType_Double, &gamma, 0.1, &gamma_min, &gamma_max, "%.4f")) {
                lopt.setGamma(gamma);
                updated = true;
            }

            const double rlrw_min = 0, rlrw_max = 100;
            static double rl_regularization_weight = lopt.getRestLengthMinimizationWeight();
            static double smoothing_weight         = lopt.getRestKappaSmoothingWeight();
            if (ImGui::DragScalar("rl_regularization_weight", ImGuiDataType_Double, &rl_regularization_weight, 0.001, &rlrw_min, &rlrw_max, "%.4f")) {
                lopt.setRestLengthMinimizationWeight(rl_regularization_weight);
                updated = true;
            }
            if (ImGui::DragScalar("smoothing_weight", ImGuiDataType_Double, &smoothing_weight, 10, &rlrw_min, &rlrw_max, "%.4f")) {
                lopt.setRestLengthMinimizationWeight(smoothing_weight);
                updated = true;
            }

            if (ImGui::InputScalar("joint pos weight", ImGuiDataType_Double, &jointPosWeight, 0, 0, "%.7f")) {
                lopt.target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(weaverLinkage, jointPosWeight, jointPosValence2Multiplier);
                lopt.invalidateAdjointState();
                updated = true;
            }
            if (ImGui::InputScalar("valence 2 multiplier", ImGuiDataType_Double, &jointPosValence2Multiplier, 0, 0, "%.7f")) {
                lopt.target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(weaverLinkage, jointPosWeight, jointPosValence2Multiplier);
                lopt.invalidateAdjointState();
                updated = true;
            }

            if (ImGui::InputScalar("linkage attraction weight", ImGuiDataType_Double, &linkage_attraction_weight, 0, 0, "%.7e")) {
                lopt.setLinkageAttractionWeight(linkage_attraction_weight);
                updated = true;
            }

            if (ImGui::Checkbox("Use centerline for surface fitting", &useCenterline)) {
                lopt.setUseCenterline(useCenterline, jointPosWeight, jointPosValence2Multiplier);
                updated = true;
            }

            if (ImGui::InputScalar("Equilibrium gradTol",  ImGuiDataType_Double, &eopts.gradTol, 0, 0, "%0.6e")) {
                lopt.setEquilibriumOptions(eopts);
            }
            ImGui::InputScalar("fd_eps",  ImGuiDataType_Double, &fd_eps, 0, 0, "%.3e");
            ImGui::InputScalar("numsteps",  ImGuiDataType_U64, &num_steps, 0, 0, "%i");
            ImGui::InputScalar("trust_region_scale",  ImGuiDataType_Double, &trust_region_scale, 0, 0, "%.7f");
            ImGui::Combo("Optimization Algorithm", (int *)(&opt_algorithm), "Newton CG\0BFGS\0\0");
            if (ImGui::Combo("Prediction Order", (int *)(&prediction_order), "Constant\0Linear\0Quadratic\0\0")) {
                lopt.prediction_order = prediction_order;
            }
            ImGui::InputScalar("Design optimization tol",  ImGuiDataType_Double, &designOptimizationTol, 0, 0, "%0.6e");

            // if (ImGui::Button("Set deployment angle", ImVec2(-1,0))) {
            //     setting_deployment_angle = true;
            // }
            if (ImGui::Checkbox("Use rest kappa", &useRestKappa)) {
                bool useRestLen = weaverLinkage.getDesignParameterConfig().restLen;
                weaverLinkage.setDesignParameterConfig(useRestLen, useRestKappa);
                lopt.setWeavingOptimization(weaverLinkage);

            }   

            if (ImGui::Checkbox("Weaving", &useWeaving)) {
                if (useWeaving) {
                    lopt.setLinkageInterleavingType(InterleavingType::weaving);
                    updated = true;
                    updateLinkageMeshes();
                    lopt.invalidateAdjointState();
                    needs_redraw = true;
                }

            }   

            if (ImGui::Checkbox("Xshell", &useXshell)) {
                if (useXshell) {
                    lopt.setLinkageInterleavingType(InterleavingType::xshell);
                    updated = true;
                    updateLinkageMeshes();
                    lopt.invalidateAdjointState();
                    needs_redraw = true;
                }

            }   

            if (ImGui::Checkbox("No Offset", &noOffset)) {
                if (noOffset) {
                    lopt.setLinkageInterleavingType(InterleavingType::noOffset);
                    updated = true;
                    updateLinkageMeshes();
                    lopt.invalidateAdjointState();
                    needs_redraw = true;
                }

            }   

            if (ImGui::Checkbox("hold closest points fixed", &holdClosestPointsFixed)) {
                weaverLinkage.set_holdClosestPointsFixed(holdClosestPointsFixed);
                updated = true;
            }   
            if (updated) stats.update(lopt);
        }

        if (ImGui::Checkbox("Show target surface", &showTargetSurface)) {
            viewer.data(target_surface_mesh_id).set_visible(showTargetSurface, view);
            viewer.data(target_surface_mesh_id).dirty = true;
        }

        ImGui::Text("J: %f",                  stats.J);
        ImGui::Text("J_fit: %e",              stats.J_target);
        ImGui::Text("||grad J||: %f",         stats.gradp_J_norm);
        ImGui::Text("J_regularization: %e",   stats.J_regularization);
        ImGui::Text("J_smoothing: %e",   stats.J_smoothing);
        ImGui::Text("E weaver: %e",         stats.E_weaver);
        ImGui::Text("Max_rod E weaver: %e", stats.E_weaver_rod_max);

        if (ImGui::Button("Save linkage data", ImVec2(-1,0))) {
            weaverLinkage.writeLinkageDebugData("weaver_opt.msh");
            std::ofstream params_out("design_parameters.txt");
            if (!params_out.is_open()) throw std::runtime_error(std::string("Couldn't open output file ") + "design_parameters.txt");
            params_out.precision(19);
            params_out << lopt.params() << std::endl;
        }

        ImGui::PopItemWidth();

        ImGui::End();

        // if (setting_deployment_angle) {
        //     // Define next window position + size
        //     ImGui::SetNextWindowPos(ImVec2(220, 10),   ImGuiSetCond_FirstUseEver);
        //     ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
        //     ImGui::Begin("Deployment", nullptr,        ImGuiWindowFlags_NoSavedSettings);

        //     ImGui::InputScalar("Angle",  ImGuiDataType_Double, &target_deployment_angle, 0, 0, "%.7f");
        //     if (ImGui::Button("Apply", ImVec2(-1,0))) {
        //         setting_deployment_angle = false;
        //         open_linkage(deployedLinkage, target_deployment_angle, eopts, lopt.getRigidMotionConstrainedJoint());
        //         lopt.getLinesearchDeployedLinkage().set(deployedLinkage);
        //         lopt.setTargetAngle(target_deployment_angle);
        //         updateLinkageMeshes();
        //         stats.update(lopt);
        //     }
        //     if (ImGui::Button("Cancel", ImVec2(-1,0))) {
        //         setting_deployment_angle = false;
        //     }

        //     ImGui::End();
        // }
    };

    viewer.launch(true, false, width, height);

    // Let the optimization finish its current iteration before exiting.
    optimization_cancelled = true;
    if (optimization_thread.joinable())
        optimization_thread.join();

    return EXIT_SUCCESS;
}
