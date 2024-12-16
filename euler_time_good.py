



import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


#Parameters: diffusivity and velocity in y direction
mu = 10000
u = -10

# --- FEM Helper Functions ---

def ref_shape_functions(xi):
    """
    Compute the reference shape functions for a triangular element.

    Parameters:
        xi: Local coordinates.

    Returns:
         Array of shape functions evaluated at the given points.
    """
    xi = np.atleast_2d(xi)
    return np.array([1 - xi[:, 0] - xi[:, 1], xi[:, 0], xi[:, 1]]).T

def derivative_of_ref_shape_functions():
    """
    Compute the derivatives of the reference shape functions with respect to the local coordinates.

    Returns: Derivatives of the reference shape functions.
    """
    return np.array([[-1, -1], [1, 0], [0, 1]])

def jacobian(node_coords):
    """
    Compute the Jacobian matrix for a triangular element.

    Parameters:
        node_coords: Coordinates of the nodes of the triangular element.

    Returns: Jacobian matrix.
    """
    dN_dxi = derivative_of_ref_shape_functions()
    return np.dot(dN_dxi.T, node_coords)

def global_x_of_xi(xi, global_node_coords):
    """
    Map a point from reference coordinates to global coordinates.

    Parameters:
        xi:  Local coordinates of the points.
        global_node_coords: Global coordinates of the element's nodes.

    Returns: Global coordinates of the points.
    """
    N = ref_shape_functions(xi)
    return np.dot(N, global_node_coords)

def det_jacobian(J):
    """
    Compute the determinant of the Jacobian matrix.

    Parameters:
        J: Jacobian matrix.

    Returns: Determinant of the Jacobian matrix.
    """
    return np.linalg.det(J)

def inverse_jacobian(J):
    """
    Compute the inverse of the Jacobian matrix.

    Parameters:
        J: Jacobian matrix.

    Returns:  Inverse of the Jacobian matrix.
    """
    return np.linalg.inv(J)

def global_deriv(J):
    """
    Compute the derivatives of shape functions in global coordinates.

    Parameters:
        J: Jacobian matrix.

    Returns: Global derivatives of the shape functions.
    """
    J_inv = inverse_jacobian(J)
    dN_dxi = derivative_of_ref_shape_functions()
    return dN_dxi @ J_inv


def mass_matrix_for_element(node_coords):
    """
    Compute the mass matrix for a triangular element using Gauss quadrature.

    Parameters:
        node_coords: Coordinates of the nodes of the triangular element.

    Returns: Mass matrix of the element.
    """
    ref_xi = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])  # Quadrature points
    mass_matrix = np.zeros((3, 3))
    J = jacobian(node_coords)  
    detJ = det_jacobian(J)  
    
    # Loop over quadrature points
    for q in range(3):
        xi = ref_xi[q, 0]
        eta = ref_xi[q, 1]
        
        # Evaluate shape functions at the quadrature point
        N = ref_shape_functions(np.array([[xi, eta]]))[0]
        
        # Integrate shape functions at quadrature points to form the mass matrix
        for i in range(3):
            for j in range(3):
                mass_matrix[i, j] += N[i] * N[j] * detJ * 1/6 
    
    return mass_matrix

def stiffness_advection(global_node_coords):
    """
     Compute the stiffness matrix with advection for a triangular element using Gauss quadrature.
    
     Parameters:
         global_node_coords: Global coordinates of the element's nodes.
    
     Returns: Stiffness matrix of the element.
     """
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]  # Quadrature points
    J = jacobian(global_node_coords)
    detJ = det_jacobian(J)
    dN_dxdy = global_deriv(J)
    N = ref_shape_functions(ref_xi)
    
    #Compute integral for the advection term
    total_advection = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            total_advection[i, j] = N[i, 0] * dN_dxdy[j, 1] + N[i, 1] * dN_dxdy[j, 1] + N[i, 2] * dN_dxdy[j, 1]
    u_term = 1/6 * total_advection * detJ
    
    #Compute diffusive term
    mu_term = 1/6 * detJ * (dN_dxdy @ dN_dxdy.T)
    
    return mu * mu_term - u * u_term

def force_vector_for_element(global_node_coords, S):
    """
    Compute the force vector for a triangular element using Gauss quadrature.

    Parameters:
        global_node_coords: Global coordinates of the element's nodes.
        S: Source term function.

    Returns: Force vector of the element.
    """
    J = jacobian(global_node_coords)
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]  # Quadrature points
    N = ref_shape_functions(ref_xi)
    global_coords = global_x_of_xi(ref_xi, global_node_coords)
    detJ = det_jacobian(J)
    
    # Construct source term for calculation
    S_vals = np.zeros((3, 1))
    for i in range(3):
        S_vals[i] += S(global_coords[i])
    
    # Calculate force vector
    return np.sum(1/6 * S_vals * N, axis=0) * detJ


def euler_step(M, K, F, Psi, dt):
    """
    Perform a single Euler step for the time integration of the system

    Parameters:
        M: Mass matrix 
        K: Stiffness matrix 
        F: Force vector 
        Psi: Solution vector
        dt: Time step

    Returns:
        Psi_new: Solution vector after one time step
    """
    # Compute dPsi/dt at the current step
    dpsidt = spsolve(M, F - K @ Psi)
    
    # Update Psi
    Psi_new = Psi + dt * dpsidt
    
    return Psi_new

def find_element_and_interpolate(global_node_coords, IEN, Psi, point):
    """
    Find the element containing the given point and compute the value of Psi at that point.

    Parameters:
        global_node_coords: Coordinates of all nodes in the global system.
        IEN: Array where each row contains the indices of the nodes defining an element.
        Psi: Array of values of Psi at the global nodes.
        point: Coordinates of the point to evaluate, given as (x, y).

    Returns:
        Psi_at_point: The interpolated value of Psi at the given point.
    """
    def is_point_in_element(global_coords, element_coords, point):
        """
        Check if a point is inside a triangular element using barycentric coordinates.
        """
        # Extract node coordinates for the element
        x1, y1 = element_coords[0]
        x2, y2 = element_coords[1]
        x3, y3 = element_coords[2]
        xp, yp = point

        # Compute areas for barycentric coordinates
        detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        alpha = ((x2 - xp) * (y3 - yp) - (x3 - xp) * (y2 - yp)) / detT
        beta = ((x3 - xp) * (y1 - yp) - (x1 - xp) * (y3 - yp)) / detT
        gamma = 1 - alpha - beta

        # Check if point lies inside the element
        return (alpha >= 0) and (beta >= 0) and (gamma >= 0), np.array([alpha, beta, gamma])

    # Loop through all elements to find the one containing the point
    for elem_idx, element in enumerate(IEN):
        element_coords = global_node_coords[element]  # Get global coordinates of nodes

        # Check if the point lies within the element
        inside, bary_coords = is_point_in_element(global_node_coords, element_coords, point)
        if inside:
            # Interpolate Psi
            element_Psi = Psi[element]  # Values of Psi at the element's nodes
            Psi_at_point = np.dot(bary_coords, element_Psi)
            return Psi_at_point
        
    


def solve_fem_euler(nodes, IEN, ID, boundary_nodes, T_max, dt, S):
    """
    Solve using Euler time-stepping.

    Parameters:
        nodes: Node coordinates.
        IEN: Element connectivity array.
        ID: Mapping of node indices to global equation indices.
        boundary_nodes: List of boundary nodes.
        T_max: Maximum simulation time.
        dt: Time step size.
        S: Source term function.

    Returns:
        psi_nodes_list: List of Psi values for each time step.
    """
    # Initialize sparse matrices
    N_equations = np.max(ID) + 1
    K = sparse.lil_matrix((N_equations, N_equations))  # Global stiffness matrix
    M = sparse.lil_matrix((N_equations, N_equations))  # Global mass matrix
    F = np.zeros(N_equations)  # Global force vector

    # Assemble loop over elements
    for e in range(IEN.shape[0]):
        node_coords = nodes[IEN[e, :], :]
        
        # Assemble the element stiffness matrix and mass matrix
        k_e = stiffness_advection(node_coords)
        m_e = mass_matrix_for_element(node_coords)

        # Assemble the global stiffness matrix and mass matrix
        for a in range(3):
            for b in range(3):
                A = ID[IEN[e, a]]
                B = ID[IEN[e, b]]
                if A >= 0 and B >= 0:
                    K[A, B] += k_e[a, b]
                    M[A, B] += m_e[a, b]

        # Assemble the force vector 
        f_e = force_vector_for_element(node_coords, S)
        for a in range(3):
            A = ID[IEN[e, a]]
            if A >= 0:
                F[A] += f_e[a]

    # Convert sparse matrices to CSR format
    K = K.tocsr()
    M = M.tocsr()

    # Initial conditions (0 everywhere)
    Psi = np.zeros(N_equations)  

    # Create list to store solution
    psi_nodes_list = []
    # Create list to store pollution values over 'Reading'
    pollution_list = []
    # Time-stepping loop 
    t = 0
    tlist = []
    while t < T_max:
        #Timestepping
        # Make fire burn for exactly 8 hours and switch off source after
        if t > 28800:
            F = 0
        else: F = F
        Psi = euler_step(M, K, F, Psi, dt)

        # Interpolate Psi values back to all nodes for visualization
        Psi_nodes = np.zeros(len(nodes))
        for i, node_id in enumerate(ID):
            if node_id >= 0:  # Ensure we only use valid node indices
                Psi_nodes[i] = Psi[node_id]
        # Append lists of solution and pollution over 'Reading'
        pollution_at_reading = find_element_and_interpolate(nodes, IEN, Psi_nodes, (440000, 171625))
        pollution_list.append(pollution_at_reading)
        psi_nodes_list.append(Psi_nodes)
        tlist.append(t)
        
        # Update time
        t += dt

        # Visualization
        if t % 1000 < dt:
            plt.triplot(nodes[:, 0], nodes[:, 1], IEN)
            plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'ro')
            plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi_nodes, shading='flat', triangles=IEN)
            plt.title(f"Time = {t:.2f} seconds")
            plt.scatter(442365, 115483, color='black')
            plt.scatter(473993, 171625, color='pink')
            plt.scatter(440000, 171625, color='orange')
            plt.colorbar()
            plt.show()
    # Plot graph of pollution over 'Reading'
    plt.plot(tlist, pollution_list)
    plt.title("Pollutant over 'Reading'")
    plt.xlabel('Time (s)')
    plt.ylabel('Pollutant')
    plt.plot()
    return psi_nodes_list






# Set up grid
nodes = np.loadtxt('griddata/las_nodes_20k.txt')
IEN = np.loadtxt('griddata/las_IEN_20k.txt', dtype=np.int64)
boundary_nodes = np.loadtxt('griddata/las_bdry_20k.txt', dtype=np.int64)
ID = np.zeros(len(nodes), dtype=np.int64)
n_eq = 0
for i in range(len(nodes[:, 1])):
    if i in boundary_nodes:
        ID[i] = -1 
    else: 
        ID[i] = n_eq
        n_eq += 1 

#Source vector 
def S_local(x, sigma=5000, mu_x=442365, mu_y=115483):
    '''
    A Gaussian source term for pollutant, centred around Southampton.
    
    Parameters:
        x : Location in grid.
        sigma : Standard deviation of the Gaussian.
        mu_x : x coordinate of Gaussian centre.
        mu_y : y coordiante of Gaussian centre.

    Returns:
    S : The computed Gaussian source term.

    '''
    # Compute the squared distance from the center (mu_x, mu_y)
    dist_squared = (x[0] - mu_x) ** 2 + (x[1] - mu_y) ** 2
    
    # Normalized Gaussian (2D)
    A = 1 / (2 * np.pi * sigma**2)  # Normalization factor
    S = A * np.exp(-dist_squared / (2 * sigma**2))  # Normalized 2D Gaussian
    return S

# Solve
psi_list = solve_fem_euler(nodes, IEN, ID, boundary_nodes, T_max=40000.0, dt=5, S=S_local)

#Print max pollution for 'Reading' and Southampton
pollution_at_reading_max = find_element_and_interpolate(nodes, IEN, psi_list[3000], (440000, 171625))
pollution_at_southampton = find_element_and_interpolate(nodes, IEN, psi_list[3000], (442365, 115483))
print(pollution_at_reading_max, pollution_at_southampton)


