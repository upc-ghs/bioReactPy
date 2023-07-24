"""
Solution of the Stokes equations using the FV-SIMPLE algorithm
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
from discretize import stokes

thresh = 1e-12

def get_discharge2D(g, u):

    """
    Calculates the flow rate on all planes normal to the flow direction.

    Inputs:
    g: the grid
    u: the cell-centre velocity field
        dimensions: (g.dim, g.num_cells)
    
    Returns:
    Q: np.array(g.Nz).
        Flow rate along the flow direction.
        One value for each element along the flow direction.
    """
    # For now only works with 'x' as main direction
    b = np.zeros(g.Nx)
    for i in np.arange(g.Nx):
        xc = g.cell_centers[0, i]
        cells = np.ravel(np.argwhere(
            ((g.cell_centers[0] > xc-1e-8) & (g.cell_centers[0] < xc+1e-8))
            ))
        b[i] = np.sum(u[cells]) * g.dx
    return b

def get_discharge3D(g, u):

    """
    Calculates the flow rate on a plane normal to the flow direction.

    Inputs:
    g: the grid
    u: the cell-centre velocity field
        dimensions: (g.dim, g.num_faces)
    
    Returns:
    b: scalar field (g.Nz).
        Flow rate along the flow direction.
        One value for each element along the flow direction.
    """

    # For now only works with 'z' as main direction
    b = np.zeros(g.Nz)
    seed = 0
    for i in np.arange(g.Nz):
        cross_section_cells = seed + np.arange(g.Nx * g.Ny)
        b[i] = np.sum(u[cross_section_cells]) * g.dx**2
        seed += g.Nx * g.Ny
    return b
 
def StokesSolver(g, d, bnd, s, p0, u0, residuals = False):

    """
    Solves Stokes flow

    At the moment, only uses SIMPLE algorithm
    but there are concrete plans of testing PISO as well

    Inputs:
    g: the grid
    d: data dictionary for fluid properties and operating conditions
    bnd: boundary conditions
    s: data dictionary for the flow solver
    p0: initial guess solution of pressure (dimensions: num_cells)
    u0: initial guess solution of velocity (dimensions: dim * num_cells)
    residuals: optional, if True, returns the list of outer residuals
    
    Returns:
    p: solution of pressure (g.num_cells)
    u: solution of velocity (g.dim, g.num_cells)
    q: solution of fluxes (g.num_faces)

    Optional
    outer: list of outer iterations
    res_u: list of residuals of velocity (in the main direction of flow)
    res_div: list of residuals of normalized continuity imbalance (div u)
    res_q: list of residuals of flow discharge
    """
    
    keyword = s["Flow solver"]

    if not residuals:
        if keyword == "Simple":
            p, u, q = SimpleAlgorithm(
                g, d, bnd, s, p0, u0, residuals
                )
        elif keyword == "Piso":
            p, u, q = PisoAlgorithm(
                g, d, bnd, s, p0, u0, residuals
                )
        return p, u, q
    else:
        if keyword == "Simple":
            p, u, q, outer, res_u, res_div, res_q = SimpleAlgorithm(
                g, d, bnd, s, p0, u0, residuals
                )
        elif keyword == "Piso":
            p, u, q, outer, res_u, res_div, res_q = PisoAlgorithm(
                g, d, bnd, s, p0, u0, residuals
                )
        return p, u, q, outer, res_u, res_div, res_q
    
def SimpleAlgorithm(g, d, bnd, s, p0, u0, residuals = False):

    """
    Solve Stokes flow using the SIMPLE algorithm by Patankar and Spalding.

    Velocity is solved using Successive Under-relaxation.
    Pressure correction is solved using a linear solver
    from the scipy.sparse librar (bicg seems the fastest).

    Solution has converged when the residuals for both velocity
    and continuity imbalance have decreased by
    'convergence_criterion' orders from the initial residuals,
    usually 3 or 4 orders is sufficient.
    

    REFERENCES

    Patankar, S. V., & Spalding, D. B. (1983).
    A calculation procedure for heat, mass and momentum transfer
    in three-dimensional parabolic flows. In Numerical prediction of flow,
    heat transfer, turbulence and combustion (pp. 54-73). Pergamon.

    Patankar, S. V. (2018).
    Numerical heat transfer and fluid flow. CRC press.

    Inputs:
    g: the grid
    d: data dictionary for fluid properties and operating conditions
    bnd: boundary conditions
    s: data dictionary for the flow solver
    p0: initial guess solution of pressure (dimensions: num_cells)
    u0: initial guess solution of velocity (dimensions: dim * num_cells)
    residuals: optional, if True, returns the list of outer residuals
    
    Returns:
    p: solution of pressure (g.num_cells)
    u: solution of velocity (g.dim, g.num_cells)
    q: solution of fluxes (g.num_faces)

    Optional
    outer: list of outer iterations
    res_u: list of residuals of velocity (in the main direction of flow)
    res_div: list of residuals of normalized continuity imbalance (div u)
    res_q: list of residuals of flow discharge
    """
    
    # Retrieve Input data
    mu = d["fluid viscosity"]

    FlowModel = s["Flow model"]
    omega_u = s["omega u"]
    omega_p = s["omega p"]
    max_iter_u = s["inner iterations u"]
    tol_u = s["inner tolerance u"]
    tol_p = s["inner tolerance p"]
    tol_continuity = s["tolerance continuity"]
    convergence_criterion = s["convergence criterion"]
    outer_iterations = s["max outer iterations"]

    # Assign boundary conditions
    u_bound = d["boundary velocity"]
    p_bound = d["boundary pressure"]

    # Initialize variables
    p = np.copy(p0)
    u = np.copy(u0)
    dp = np.zeros(g.num_cells)

    dp_bound = np.zeros(g.num_faces)

    # Assume homogeneous viscosity
    visc = mu * np.ones(g.num_cells)

    # Calculate micro-continuum permeability
    if FlowModel == 'microcontinuum':
        initial_permeability = d["initial permeability"]
        phi = d["porosity field"]
        num = np.power(np.ones(g.num_cells) - phi, 2)
        den = np.power(phi, 3)
        inv_perm = 1 / initial_permeability * np.divide(num, den)
        visc = mu/phi

    #Discretize momentum equations
    Au, Au_bound = stokes.discretize_momentum(
        g, bnd, visc, u_bound[0], 'x', FlowModel
        )
    Av, Av_bound = stokes.discretize_momentum(
        g, bnd, visc, u_bound[1], 'y', FlowModel
        )

    if FlowModel == 'microcontinuum':
        Au += sps.diags(mu * inv_perm * g.cell_volumes)
        Av += sps.diags(mu * inv_perm * g.cell_volumes)

    acu = Au.diagonal()
    acv = Av.diagonal()

    ac = np.vstack((acu, acv))

    if g.dim > 2:
        Aw, Aw_bound = stokes.discretize_momentum(
            g, bnd, visc, u_bound[2], 'z', FlowModel
            )
        if FlowModel == 'microcontinuum':
            Aw += sps.diags(mu * inv_perm * g.cell_volumes)
        acw = Aw.diagonal()
        ac = np.vstack((ac, acw))

    # Discretize pressure correction equation
    Ap = stokes.discretize_pressure_correction(
        g, bnd, ac, FlowModel
        )
   
    #Initialize monitors
    initial_div_u = 0
    initial_res_u = 0

    iters = []
    residual_u = []
    residual_divu = []
    residual_q = []

    outer = 0

    #Run Simple algorithm

    for i in np.arange(outer_iterations):

        outer = i
        if residuals:
            print('Simple outer iter', i)

        # Step 1 - Calculate pressure gradient
        p_faces = stokes.get_face_pressure(g, bnd, p, p_bound)
        gradp = stokes.discretize_pressure_gradient(g, p_faces)
        if FlowModel == 'NoSlip':
            gradp[:,bnd.list_wall] = 0

        # Step 2 - Solve momentum
        b = Au_bound -  gradp[0]
        u[0], res_u, iter_u = sor_solver(
            Au, b, omega_u, u[0], tol_u, max_iter_u
            )
        if residuals:
            print('momentum x', iter_u, res_u)

        b = Av_bound -  gradp[1]
        u[1], res_v, iter_v = sor_solver(
            Av, b, omega_u, u[1], tol_u, max_iter_u
            )
        if residuals:
            print('momentum y', iter_v, res_v)

        if g.dim > 2:
            b = Aw_bound -  gradp[2]
            u[2], res_u, iter_u = sor_solver(
                Aw, b, omega_u, u[2], tol_u, max_iter_u
                )
            if residuals:
                print('momentum z', iter_u, res_u)

        if FlowModel == 'NoSlip' and len(bnd.list_wall) > 0:
            assert np.all(u[:, bnd.list_wall]) == 0
            
        # Step 3 - Check mass conservation
        if g.dim == 2:
            q = get_discharge2D(g, u[0])
        else:
            q = get_discharge3D(g, u[2])
            
        qm = np.average(q)
        q_in = q[0]
        q_out = q[-1]
        eq = abs(q_out-qm)/qm
        #print('q_in', q_in, 'q_out', q_out, 'qm', qm)
        if residuals:
            print('Discharge error', eq)
        
        # Step 4 - Retrieve fluxes
        q_faces = stokes.get_face_fluxes(
            g, bnd, u, ac, p, u_bound, p_faces
            )

        # Step 5 - Calculate continuity imbalance
        div_u, div_q = stokes.discretize_continuity(g, q_faces)

        # Step 6 - Check convergence
        # Normalize the velocity divergence using the average discharge
        norm_div_u = abs(np.sum(div_q)/g.num_cells/qm)
        if residuals:
            print('continuity imbalance', norm_div_u)
        if i == 0:
            initial_res_u = res_u
            initial_div_u = norm_div_u

        # Calculate residual
        res_div_u = norm_div_u/initial_div_u

        # Print monitor residuals
        if residuals:
            iters.append(i)
            residual_u.append(res_u)
            residual_q.append(eq)
            residual_divu.append(norm_div_u)

        # Convergence criterion
        if (
            (np.logical_or(
                norm_div_u < tol_continuity,
                res_div_u < convergence_criterion
                )
             )
            and
            (res_u/initial_res_u < convergence_criterion)
            ):
            print('Convergence met, outer iteration', i)
            print('Continuity imbalance', norm_div_u)
            break

        # Step 7 - Solve pressure correction
        # Apparently dividing both terms by face areas
        # makes the linear solver much faster in that
        # a value of tol_p of 1e-2 is sufficient for convergence.
        # Instead, bringing the face areas
        # to the left hand side requires lower tol_p for convergence.
        # Not really sure why but it must have to do with how
        # the direct solver deals with the A matrix
        b = div_u / g.dx**(g.dim-1)

        # Among all solvers, bicgstab seems to be the fastest
        # for increasing grid resolution
        if g.dim < 3:
            dp = scipy.sparse.linalg.spsolve(Ap, b)

        else:        
            dp, exitCode = scipy.sparse.linalg.bicg(Ap, b, tol=tol_p)
            if exitCode != 0:
                print('failed bicg')
                assert False


        # Step 8 - Update flow field
        # Update pressure
        p += omega_p * dp
        if FlowModel == 'NoSlip' and len(bnd.list_wall) > 0:
            assert np.all(p[bnd.list_wall]) == 0

        # Correct velocity using the pressure correction gradient
        dp_faces = stokes.get_face_pressure(g, bnd, dp, dp_bound)
        graddp = stokes.discretize_pressure_gradient(g, dp_faces)
        if FlowModel == 'NoSlip':
            graddp[:,bnd.list_wall] = 0

        u -= graddp / ac

    if not residuals:
        return p, u, q_faces
    else:
        return p, u, q_faces, iters, residual_u, residual_divu, residual_q

def sor_solver(
    A, b, omega, initial_guess, convergence_criteria, iterations
    ):

    """
    Inputs:
    A: nc x nc scipy sparse matrix
    b: nc dimensional numpy vector
    omega: relaxation factor
    initial_guess: An initial solution guess
    convergence_criteria: tolerance
    iterations: max number of iterations

    Returns:
    phi: np.array(g.num_cells) solution vector
    residual: the last residual
    count: the last iteration counter
    """

    phi = np.copy(initial_guess)
    initial_residual = np.linalg.norm(A*phi - b)
    ac = A.diagonal()
  
    count = 0

    while count < iterations:

        H = A * phi - ac * phi

        phi = (1 - omega) * phi + omega / ac * (b - H)

        count += 1
        residual = np.linalg.norm(A*phi - b)          
        if residual/(initial_residual+thresh) < convergence_criteria:
            break

    return phi, residual, count
