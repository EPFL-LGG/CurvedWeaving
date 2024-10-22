from finite_diff import gradient_convergence_plot
from scipy.sparse import csc_matrix
import numpy as np
import numpy.linalg as la
from linkage_optimization import OptEnergyType
import matplotlib.pyplot as plt

def get_matvec_error_for_linkage(linkage):
    hessian = linkage.hessianPerSegmentRestlen()
    hessian.reflectUpperTriangle()
    hessian = hessian.compressedColumn()

    n_dof = linkage.numExtendedDoFPSRL()
    perturb = np.random.uniform(-1, 1, n_dof)
    input_vector = perturb
    code_output = linkage.applyHessianPerSegmentRestlen(input_vector)
    matrix_output = hessian * input_vector
    error = la.norm(code_output - matrix_output) / la.norm(code_output)
    print("The hessian vector multiplication error is: ", error)

def get_J_fd(optimizer, fd_eps, direction, etype = OptEnergyType.Full):
    curr_params = optimizer.params()
    Jplus = optimizer.J(curr_params + direction * fd_eps, etype)
    Jminus = optimizer.J(curr_params - direction * fd_eps, etype)
    return (Jplus - Jminus) / (2 * fd_eps)

def get_J_fd_error(optimizer, fd_eps, direction, etype = OptEnergyType.Full):
    grad_J = optimizer.gradp_J(optimizer.params(), etype)
    fd_J = get_J_fd(optimizer, fd_eps, direction, etype)
    fd_error = np.abs((np.dot(grad_J, direction) - fd_J) / np.dot(grad_J, direction))
    return fd_error

def gradient_convergence_plot(optimizer, direction, energyTypeString, energyType):
    curr_params = optimizer.params()
    grad_J = optimizer.gradp_J(curr_params, energyType);
    an_delta_J = np.dot(grad_J, direction)
    epsilons = np.logspace(np.log10(1e-11), np.log10(1e-2), 40)
    errors = []
    for eps in epsilons:
        fd_J = get_J_fd(optimizer, eps, direction, energyType)
        err = np.abs((an_delta_J - fd_J) / an_delta_J)
        errors.append(err)
    plt.title('Directional derivative fd test for gradient - {}'.format(energyTypeString))
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.savefig('gradient_{}_validation_design_opt_after_optimize.png'.format(energyTypeString), dpi = 300)

def get_J_fd_hessian_error(optimizer, fd_eps, direction, etype = OptEnergyType.Full):
    curr_params = optimizer.params()
    gJplus = optimizer.gradp_J(curr_params + direction * fd_eps, etype)
    gJminus = optimizer.gradp_J(curr_params - direction * fd_eps, etype)
    fd_J_hessian = (gJplus - gJminus) / (2 * fd_eps)
    hv_J = optimizer.apply_hess_J(curr_params, direction, etype)
    fd_error = la.norm(hv_J - fd_J_hessian) / la.norm(hv_J)
    return fd_error

def hessian_convergence_plot(optimizer, direction, energyTypeString, energyType):
    epsilons = np.logspace(np.log10(1e-11), np.log10(1e-2), 40)
    errors = []
    for eps in epsilons:
        err = get_J_fd_hessian_error(optimizer, eps, direction, energyType)
        errors.append(err)
    plt.title('Directional derivative fd test for Hessian - {}'.format(energyTypeString))
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.savefig('hessian_{}_validation_design_opt_after_optimize.png'.format(energyTypeString), dpi = 300)
