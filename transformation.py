#%% Import Libraries

from pymatgen.transformations.standard_transformations import RotationTransformation,SupercellTransformation
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.structure import Structure
import plotly.graph_objs as go
from glob import glob 
import numpy as np
import logging
from vector import unit,rotate_vectors,get_angle
from plotting import plot_coords,plot_structure,plot_shapes,plot_vectors_3d,plot_points,plot_lattice,plot_planes,plot_supercell_boundaries
import pandas as pd

# Define logging configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:%(message)s')

file_handler = logging.FileHandler('transformation.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# %% Coordinate Transformations
def get_miller_to_coord(struct,miller):
    return struct.lattice.matrix[0]*miller[0]+struct.lattice.matrix[1]*miller[1]+struct.lattice.matrix[2]*miller[2]

def coord_to_miller(struct,coord):
    return [coord[0]/struct.lattice.a,coord[1]/struct.lattice.b,coord[2]/struct.lattice.c]

def find_rotation_axis(struct,miller=[1,1,1],coord_axis=[0,0,1]):

    #vector representing chain axis 
    chain_vec = get_miller_to_coord(struct=struct, miller=miller)

    #axis of rotation for bringing chain to z axis 
    rot_vec = np.cross(chain_vec,coord_axis)
    rot_axis = unit(rot_vec)

    return rot_axis

def rotate_lattice(struct,angle,deg=True,coord_axis=[0,0,1],miller = [1,1,1]):
    
    rot_axis = find_rotation_axis(struct,miller,coord_axis)
    logger.info(f'Rotation axis: {rot_axis}')
    

    #find angle between chain and z-axis
    chain_vector = get_miller_to_coord(struct=struct,miller=miller)
    chain_angle = get_angle(chain_vector,coord_axis,deg=deg)
    
    logger.info(f'Rotating by {chain_angle-angle}.')

    rotate = RotationTransformation(axis = rot_axis,
                                    angle = chain_angle - angle,
                                    angle_in_radians = not deg)
    
    struct_rot = rotate.apply_transformation(struct)

    new_axes = rotate_vectors([[1,0,0],[0,1,0],[0,0,1]],rot_axis,chain_angle - angle)

    return struct_rot,new_axes

# Generate an n x n x n lattice of the primary atom using the rhombohedral (original)
# lattice vectors
def generate_lattice(n, Rlatvec, patom=[0, 0, 0]):
    # Define lattice vectors a, b, c
    a,b,c = Rlatvec
    logger.debug('Original lattice vectors:',a,b,c)

    # Initialize lattice data array
    pdat = np.zeros((n**3, 3))

    # Initialize primary atom
    pdat[0] = patom
    logger.debug('Primary atom added: ',patom)

    # Replicate the primary atom along a-axis
    for i in range(1, n, 2):
        pdat[i] = ((i+1) / 2) * a
        pdat[i + 1] = -1 * pdat[i]
    logger.debug('Replicated atoms along a-axis')
    logger.debug('Lattice points added: ',pdat[:i+2])
    
    
    # Replicate the a-axis array along b-axis
    r = 1
    for i in range(n-1, n ** 2 - 2 * n, 2 * n):
        pdat[i + 1:i + n + 1] = pdat[:n] + r * np.tile(b, (n, 1))
        pdat[i + n + 1:i + 2 * n + 1] = -1 * pdat[i + 1:i + n + 1]
        r += 1
    logger.debug('Replicated atoms along b-axis')
    

    # Replicate the ab-matrix along c-axis
    r = 1
    for i in range(n ** 2-1, n ** 3 - 2 * n ** 2, 2 * n ** 2):
        pdat[i + 1:i + n ** 2 + 1] = pdat[:n ** 2] + r * np.tile(c, (n ** 2, 1))
        pdat[i + n ** 2 + 1:i + 2 * n ** 2 + 1] = -1 * pdat[i + 1:i + n ** 2 + 1]
        r += 1
    logger.debug('Replicated atoms along c-axis')
    

    return np.array(remove_duplicates(pdat))


# Finds the best lattice vectors for the cuboidal cell
def FindUnitCell(struct_rot,tol,n):
    """
    Finds the best lattice vectors for the cuboidal cell, given the rotated structure and tolerance value.
        struct_rot: Rotated structure (pymatgen.core.structure.Structure)
        tol: Tolerance value for the dot product of the unit vectors (float)
        n: Number of lattice points along each direction (int)
    Returns:
        BestLatDim: Best lattice dimensions (numpy.ndarray)
        BestLatVect: Best lattice vectors (numpy.ndarray)
        (alfa, beta, gama): Angles of the cuboidal cell (tuple)
    """
    #check if n is odd and throw error if not
    if n%2 == 0:
        raise ValueError(f'n must be odd, got {n}')
    
    #initialize x,y, and z bounds 
    x_high,y_high,z_high = [float("inf"),float("inf"),float("inf")]

    #generate n lattice points for the rotated structure 
    rotated_lattice_points = generate_lattice(n,struct_rot.lattice.matrix)
    
    for point in rotated_lattice_points:
        #turn point into vector 
        position_vector = unit(np.array(point))

        #check if point lies on x-axis
        x_angle = abs(get_angle(position_vector,[1,0,0],deg=False))
        y_angle = abs(get_angle(position_vector,[0,1,0],deg=False))
        z_angle = abs(get_angle(position_vector,[0,0,1],deg=False))

        if x_angle <= tol and point[0] < x_high:
            x_high = point[0]
            x_point = point
        
        if y_angle <= tol and point[1] < y_high:
            y_high = point[1]
            y_point = point
        
        if z_angle <= tol and point[2] < z_high:
            z_high = point[2]
            z_point = point
    
    #check to see if candidate points were found
    try:
        is_defined = len(x_point) > 0 and len(z_point) and len(y_point) >0
    except:
        raise ValueError(f'Could not find lattice vectors for the cuboidal cell of size: {n}')

    # Write the new lattice coordinates to the input file
    BestLatVect = np.vstack([x_point,y_point,z_point])

    #find length of lattice vectors 
    mag_x = np.linalg.norm(x_point)
    mag_y = np.linalg.norm(y_point)
    mag_z = np.linalg.norm(z_point)

    # Store the minimum values in BestLatDim
    BestLatDim = np.array([mag_x, mag_y, mag_z])

    # Display results
    gama = np.arccos(np.dot(x_point, y_point) / (np.linalg.norm(x_point) * np.linalg.norm(y_point))) * 180 / np.pi
    alfa = np.arccos(np.dot(y_point, z_point) / (np.linalg.norm(y_point) * np.linalg.norm(z_point))) * 180 / np.pi
    beta = np.arccos(np.dot(x_point, z_point) / (np.linalg.norm(x_point) * np.linalg.norm(z_point))) * 180 / np.pi
    logger.info('Cell angles:')
    logger.info(f'alfa: {alfa:.15f} beta: {beta:.15f} gama: {gama:.15f}')
    logger.info('Best Lat Dims:')
    logger.info(f'x: {BestLatDim[0]:.15f} y: {BestLatDim[1]:.15f} z: {BestLatDim[2]:.15f}')
    logger.info('Best Lattice vectors:')
    logger.info(BestLatVect)

    return BestLatDim,BestLatVect,(alfa,beta,gama)

def find_n(struct,angle,max_n=205,tol=0.001):
    struct_rot,new_axes = rotate_lattice(struct,angle)
    
    logger.info(f'Angle of chain with respect to z-axis: {angle}º')
 
    for i in range(3,max_n,11):
        try:
            _,_,_ = FindUnitCell(struct_rot,n=i,tol=tol)
            for ii in range(i-10,i+1):
                try: 
                    _,_,_ = FindUnitCell(struct_rot,n=ii,tol=tol)
                    n=ii
                    logger.info(f'Best n value for {angle}º: {ii}')
                    break
                except Exception as e:
                    logger.info(f'Failed to find best n value for {ii}')
                    logger.info(e)
            break
        except Exception as e:
            logger.warning(f'Failed to find best n value for {i} with max n: {max_n}')
            logger.warning(e)
            continue
            return None
    
    #check if n is odd
    if n%2 == 0:
        n+=1
    return n

def lattice_vectors(a, b, c, alpha, beta, gamma):
    # Convert angles from degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Lattice vector a
    vec_a = np.array([a, 0, 0])

    # Lattice vector b
    vec_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

    # Lattice vector c
    vec_c_x = c * np.cos(beta)
    vec_c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    vec_c_z = c * np.sqrt(1 - np.cos(beta)**2 - (np.cos(alpha) - np.cos(beta) * np.cos(gamma))**2 / np.sin(gamma)**2)
    vec_c = np.array([vec_c_x, vec_c_y, vec_c_z])

    return vec_a, vec_b, vec_c

def remove_atoms(supercell_rhom,new_origin,BestLatVect):
    r_a,r_b,r_c = BestLatVect

    low_point = new_origin 
    high_point = new_origin + r_a+r_b+r_c

    N_low_x = np.cross(r_b,r_c)/np.linalg.norm(np.cross(r_b,r_c))
    N_low_y = np.cross(r_c,r_a)/np.linalg.norm(np.cross(r_c,r_a))
    N_low_z = np.cross(r_a,r_b)/np.linalg.norm(np.cross(r_a,r_b)) 

    N_high_x = -N_low_x
    N_high_y = -N_low_y
    N_high_z = -N_low_z    
    
    coords = supercell_rhom.cart_coords

    keep_idx = []
    remove_idx = []

    for i in range(len(supercell_rhom)):
        #check each coord and see if its within bounds
        x,y,z = coords[i]

        v_low = coords[i] - low_point
        v_high = coords[i] - high_point

        inside_x = np.dot(N_low_x,v_low) >= 0 and np.dot(N_high_x,v_high) >= 0
        inside_y = np.dot(N_low_y,v_low) >= 0 and np.dot(N_high_y,v_high) >= 0
        inside_z = np.dot(N_low_z,v_low) >= 0 and np.dot(N_high_z,v_high) >= 0

        if not (inside_x and inside_y and inside_z):
            #atom is not within bounds, remove from structure
            remove_idx.append(i)
            
        else:
            keep_idx.append(i)
            supercell_rhom[i].coords = coords[i]-new_origin

    copied_sites = [supercell_rhom[i] for i in keep_idx]
    rect_unitcel = Structure.from_sites(copied_sites)

    rect_lattice = np.array([r_a,r_b,r_c])

    rect_unit_cell = Structure(rect_lattice,rect_unitcel.species,rect_unitcel.cart_coords,coords_are_cartesian=True)

    return rect_unit_cell

def remove_duplicates(coords):
    """
    Remove duplicate coordinates from a list.
    
    Parameters:
    coords (list of lists or tuples): A list of 3D coordinates.
    
    Returns:
    list: A new list with duplicate coordinates removed.
    """
    # Use a set to track seen coordinates
    seen = set()
    unique_coords = []
    
    for coord in coords:
        # Convert the coordinate to a tuple (since lists are not hashable)
        coord_tuple = tuple(coord)
        if coord_tuple not in seen:
            seen.add(coord_tuple)
            unique_coords.append(coord)
    
    return unique_coords

def remove_lattice_points(lattice_points,rect_lattice,new_origin=[0,0,0]):
    r_a,r_b,r_c = rect_lattice
    
    low_point = new_origin 
    high_point = new_origin + r_a+r_b+r_c

    N_low_x = np.cross(r_b,r_c)/np.linalg.norm(np.cross(r_b,r_c))
    N_low_y = np.cross(r_c,r_a)/np.linalg.norm(np.cross(r_c,r_a))
    N_low_z = np.cross(r_a,r_b)/np.linalg.norm(np.cross(r_a,r_b)) 

    N_high_x = -N_low_x
    N_high_y = -N_low_y
    N_high_z = -N_low_z    
    
    coords = lattice_points

    keep_idx = []
    remove_idx = []

    for i in range(len(lattice_points)):
        #check each coord and see if its within bounds
        x,y,z = coords[i]

        v_low = coords[i] - low_point
        v_high = coords[i] - high_point

        inside_x = np.dot(N_low_x,v_low) >= 0 and np.dot(N_high_x,v_high) >= 0
        inside_y = np.dot(N_low_y,v_low) >= 0 and np.dot(N_high_y,v_high) >= 0
        inside_z = np.dot(N_low_z,v_low) >= 0 and np.dot(N_high_z,v_high) >= 0

        if not (inside_x and inside_y and inside_z):
            #atom is not within bounds, remove from structure
            remove_idx.append(i)
            
        else:
            keep_idx.append(i)
            

    copied_sites = [lattice_points[i] for i in keep_idx]
    

    return copied_sites

def move_origin(supercell_rhom,rect_domain):
    #find center of cuboidal domain by moving halfway in each cuboidal lattice direction
    r_a,r_b,r_c = rect_domain
    rect_center = r_a/2 + r_b/2 + r_c/2

    #lattice vectors of supercell
    s_a,s_b,s_c = supercell_rhom.lattice.matrix

    #unit vectors of rhombohedral sueprcell
    u_a = s_a/np.linalg.norm(s_a)
    u_b = s_b/np.linalg.norm(s_b)
    u_c = s_c/np.linalg.norm(s_c)

    #find center of supercell by moving half way in each lattice direction
    supercell_center = s_a/2 + s_b/2 + s_c/2

    #find displacement vector
    displacement = supercell_center - rect_center

    #get inv transformation matrix 

    # inverse transformation matrix such that col 1 = a, col 2 = b, col 3 = c
    inv_T_mat = np.column_stack((u_a,u_b,u_c))
    T_mat = np.linalg.inv(inv_T_mat)

    #transform displaced origin point 
    displaced_point = np.array([0,0,0]) + displacement

    #transform to supercell coordinates
    supercell_coords = T_mat@displaced_point

    #snap to nearest lattice point
    snap_coords = np.round(supercell_coords)

    #transform back to original coordinates
    new_origin = inv_T_mat@snap_coords

    return new_origin



def generate_cuboidal_unit_cell(path_to_data,angle,tol=0.0001,n=None,plot_final=False):
    # 1. Read in unit cell structure from data file
    
    struct = LammpsData.from_file(path_to_data,atom_style='atomic').structure
    

    # 2. Get lattice vectors from datafile
    RLatvec = struct.lattice.matrix
    
    # 3. Get Best n and lattice vectors for cuboidal cell
    if n is None:
        n = find_n(struct,angle,tol=tol) 
    logger.info(f'Number of copies to make of lattice for search: {n}')

    struct_rot,new_axes = rotate_lattice(struct,angle)

    BestLatdims,BestLatvect,angles = FindUnitCell(struct_rot,n=n,tol=tol)
    r_a,r_b,r_c = BestLatvect

    rect_domain = [r_a,r_b,r_c]
    # fig1 = plot_shapes([struct_rot.lattice.matrix,np.array(rect_domain)])

    #determine the amount of scaling necessary for each lattice vector
    #using n for now 
    supercell_rhom = struct_rot.make_supercell(n,in_place=False)

    # plot_structure(supercell_rhom)

    # find box bounds such that box_low = [x_low,y_low,z_low] and box_high = [x_high,y_high,z_high] are contained inside supercell 

    new_origin = move_origin(supercell_rhom,rect_domain)

    # fig = plot_shapes([supercell_rhom.lattice.matrix,np.array(rect_domain),np.array(rect_domain)],origin_list=[[0,0,0],[0,0,0],new_origin])
    # rect_unit_cell,rect = remove_atoms(supercell_rhom,new_origin,BestLatdims)
    rect_unit_cell = remove_atoms(supercell_rhom,new_origin,BestLatvect)
    if plot_final:
        fig = plot_structure(rect_unit_cell,atom_size=3)
        fig.update_layout(scene_camera=dict(projection=dict(type='orthographic')))
        return rect_unit_cell,fig
    else:
        
        return rect_unit_cell
            

def replicate_coords(coords, axis_x, axis_y, axis_z, n):
    # Ensure inputs are numpy arrays
    coords = np.array(coords)
    axis_x = np.array(axis_x)
    axis_y = np.array(axis_y)
    axis_z = np.array(axis_z)
    
    # Create an empty list to store all replicated coordinates
    replicated_coords = []
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Calculate the offset for current replication
                offset = i * axis_x + j * axis_y + k * axis_z
                # Apply the offset to each coordinate
                new_coords = coords + offset
                # Store the new coordinates
                replicated_coords.append(new_coords)
    
    # Convert the list to a numpy array
    replicated_coords = np.array(replicated_coords)
    
    # Reshape to have original coordinate sets along one axis
    replicated_coords = replicated_coords.reshape(-1, coords.shape[0], coords.shape[1])
    
    return replicated_coords

# %%
if __name__ == '__main__':
    fs = glob('/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/atom_labels/data.*')
    df = {}
    thetas = [0]
    dim = [50,50,150]

    for theta in thetas:
        max_x,max_y,max_z = [0,0,0]
        for f in fs:
            polytype = f.split('.')[-1]
            df[polytype] = generate_cuboidal_unit_cell(f,angle=theta,tol=0.1)

            if df[polytype].lattice.lengths[0] > max_x:
                max_x = df[polytype].lattice.lengths[0]
            if df[polytype].lattice.lengths[1] > max_y:
                max_y = df[polytype].lattice.lengths[1]
            if df[polytype].lattice.lengths[2] > max_z:
                max_z = df[polytype].lattice.lengths[2]

        #get scaling for each direction based on largest side length 
        if dim[0] > max_x:
            max_x = dim[0]
        if dim[1] > max_y:
            max_y = dim[1]
        if dim[2] > max_z:
            max_z = dim[2]

        scale = {}
        for polytype in df:
            lx,ly,lz = df[polytype].lattice.lengths
            nx = np.ceil(max_x/lx)
            ny = np.ceil(max_y/ly)
            nz = np.ceil(max_z/lz)

            scale[polytype] = [nx,ny,nz]

                
        super_cells = {}
        for polytype in df:
            super_cells[polytype] = df[polytype].make_supercell(scale[polytype],in_place=False)

            lammps_data = LammpsData.from_structure(super_cells[polytype],atom_style='charge')
            filename = f'/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/rect_supercells/orientation/50-50-150/data.{polytype}_{theta}_charge'
            lammps_data.write_file(filename)
#%%
            




# %%
