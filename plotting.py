#%%
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import numpy as np
import os
# %% Define logging configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:%(message)s')

if not os.path.isdir('logs'):
    os.makedirs('logs')
file_handler = logging.FileHandler('logs/plotting.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

#%%
def plot_supercell_boundaries(struct,n,colors=None,fig=None):
    if fig is None:
        fig = go.Figure()

    struct_list = []
    origin_list = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                origin = a*struct.lattice.matrix[0] + b*struct.lattice.matrix[1] +c*struct.lattice.matrix[2]
                origin_list.append(origin)
                struct_list.append(struct)
    
    f = plot_shapes([i.lattice.matrix for i in struct_list],origin_list=[i for i in origin_list],fig=fig,colors=colors,width=1)
    return f

def plot_planes(normals, points, plane_size=10,fig=None):
    """
    Plots a series of planes given their unit normal vectors and points on the planes.

    Parameters:
    - normals: List of normal vectors (each a list or numpy array of 3 elements).
    - points: List of points (each a list or numpy array of 3 elements) where each plane passes through.
    - plane_size: Size of the plane to plot (extends from -plane_size to plane_size in both directions from the point).

    Returns:
    - fig: Plotly figure object.
    """
    if len(normals) != len(points):
        raise ValueError("The length of normals and points must be the same.")

    # Initialize the plot
    if fig is None:
        fig = go.Figure()

    for normal, point in zip(normals, points):
        # Calculate two vectors on the plane
        if np.allclose(normal, [0, 0, 1]):  # special case to avoid zero vector cross product
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.cross(normal, [0, 0, 1])
        v1 /= np.linalg.norm(v1)

        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2)

        # Generate a grid of points on the plane
        plane_x, plane_y = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                                       np.linspace(-plane_size, plane_size, 10))
        plane_z = np.zeros_like(plane_x)

        # Calculate the actual coordinates of the plane
        plane_coords = point + v1 * plane_x[..., np.newaxis] + v2 * plane_y[..., np.newaxis]
        
        # Extract coordinates
        x = plane_coords[:, :, 0]
        y = plane_coords[:, :, 1]
        z = plane_coords[:, :, 2]

        # Add plane to the plot
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.7))

    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        ),
        title="Planes in 3D Space"
    )

    return fig

def plot_points(points,fig=None):
    if fig is None:
        fig = go.Figure()

    for point in points:
        fig.add_trace(go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name='Point'
            ))
            
    fig.update_layout(scene=dict(aspectmode='data'))
    return fig

def plot_shapes(matrices,origin_list=None,fig=None,colors=None,width=2):
    if fig is None:
        fig = go.Figure()
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan', 'magenta']  # Define a list of colors
    if colors == 'black':
        colors = ['black' for i in range(len(matrices))]
    for index, matrix in enumerate(matrices):
        
        # Ensure the input matrix is 3x3
        assert matrix.shape == (3, 3), "Each input matrix must be 3x3."

        # Extract the vectors from the matrix
        v0, v1, v2 = matrix

        # Define the vertices based on the vectors
        if origin_list is None:
            origin = np.array([0, 0, 0])
        else:
            origin = origin_list[index]
        
        vertices = origin+np.array([
            [0,0,0],
            v0,
            v1,
            v2,
            v0 + v1,
            v1 + v2,
            v2 + v0,
            v0 + v1 + v2
        ])

        # Define the edges of the shape
        edges = [
            [0, 1], [0, 2], [0, 3],
            [1, 4], [1, 6], [2, 4],
            [2, 5], [3, 5], [3, 6],
            [4, 7], [5, 7], [6, 7]
        ]

        # Create the wireframe lines
        lines = []
        for edge in edges:
            start, end = edge
            lines.append([vertices[start], vertices[end]])

        # Prepare data for plotting
        x_lines = []
        y_lines = []
        z_lines = []
        
        for line in lines:
            x_lines.extend([line[0][0], line[1][0], None])  # x coordinates
            y_lines.extend([line[0][1], line[1][1], None])  # y coordinates
            z_lines.extend([line[0][2], line[1][2], None])  # z coordinates

        # Add the wireframe trace to the figure with a unique color
        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color=colors[index % len(colors)], width=width),
            name=f'Shape {index + 1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        ),
        title="3D Shapes Outlines Defined by Matrices"
    )
    fig.update_layout(scene=dict(aspectmode='data'))

    


    fig.show()
    return fig

def plot_lattice(pdats,op=1,size=5,fig=None):
    if fig is None:
        fig = go.Figure()
    for pdat in pdats:
        fig.add_trace(go.Scatter3d(x=pdat[:,0], y=pdat[:,1], z=pdat[:,2], mode='markers', opacity= op,marker=dict(size=size)))
        fig.update_layout(
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='data'))
    
    return fig
    
    
def plot_vectors_3d(vectors, colors=None, labels=None, title=None,fig=None):

    """
    Create a 3D scatter plot with multiple vectors, each ending with an arrow.

    Parameters:
        vectors (list of lists or array-like): List or array containing vectors.
                                               Each vector should be a list or array with six elements:
                                               [x_start, y_start, z_start, dx, dy, dz],
                                               where (x_start, y_start, z_start) are the coordinates of the starting point
                                               of the vector, and (dx, dy, dz) are the components of the vector.
        colors (list of str, optional): List of colors for each vector. If not provided, defaults to 'blue'.
        labels (list of str, optional): List of labels for each vector. If not provided, no labels are added.
        title (str, optional): Title of the plot. If not provided, no title is added.

    Returns:
        None
    """
    # Define layout
    layout = go.Layout(scene=dict(xaxis=dict(title='X'),
                                  yaxis=dict(title='Y'),
                                  zaxis=dict(title='Z')),
                       margin=dict(l=0, r=0, t=0, b=0),
                       title=title)

    # Create figure object with layout
    if fig is None:
        fig = go.Figure(layout=layout)
    
    x_start, y_start, z_start = [0,0,0]
    # Add vectors
    for i, vector in enumerate(vectors):
        dx, dy, dz = vector

        color = colors[i] if colors else 'blue'

        # Add vector line
        fig.add_trace(go.Scatter3d(x=[x_start, x_start + dx],
                                    y=[y_start, y_start + dy],
                                    z=[z_start, z_start + dz],
                                    mode='lines',
                                    line=dict(color=color, width=3),
                                    name=labels[i] if labels else None))

        # Add arrow
        arrow_len = 0.1 * (dx**2 + dy**2 + dz**2)**0.5  # Length of arrow proportional to vector length
        fig.add_trace(go.Cone(x=[x_start + dx], y=[y_start + dy], z=[z_start + dz],
                               u=[dx], v=[dy], w=[dz],
                               colorscale=[[0, color], [1, color]],
                               sizemode="absolute", sizeref=arrow_len, showscale=False))

        # Add label
        label_x = [x_start + dx / 2]
        label_y = [y_start + dy / 2]
        label_z = [z_start + dz / 2]
        fig.add_trace(go.Scatter3d(x=label_x, y=label_y, z=label_z,
                                   mode='text',
                                   text=labels[i] if labels else None,
                                   textfont=dict(color=color, size=12),
                                   showlegend=False))

    # Show plot

    return fig
    

def plot_structure(structure, coord_axes=None, atom_size=15, plot_lattice=True, miller_indices_vector=None,fig=None):
    """
    Plot atomic coordinates of a structure using plotly, with options to plot lattice vectors and a Miller indices vector.

    Args:
        structure (pymatgen Structure): The structure object.
        atom_size (float): Size of atoms in the plot.
        plot_lattice (bool): Whether to plot lattice vectors.
        miller_indices_vector (tuple): Miller indices vector to plot.

    Returns:
        None
    """
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Extract atomic coordinates and species
    atomic_coordinates = structure.cart_coords
    atomic_species = structure.species

    # Extract x, y, and z coordinates
    x_coords = atomic_coordinates[:, 0]
    y_coords = atomic_coordinates[:, 1]
    z_coords = atomic_coordinates[:, 2]

    # Get unique species and assign colors from predefined palette
    species_set = sorted(list(set(atomic_species)))
    color_map = {species: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, species in enumerate(species_set)}
    colors = [color_map[s] for s in atomic_species]

    # Create scatter plot for atoms
    if fig is None:
        fig = go.Figure()

    # Add traces for each species
    for species, color in color_map.items():
        mask = [s == species for s in atomic_species]
        fig.add_trace(go.Scatter3d(x=x_coords[mask], y=y_coords[mask], z=z_coords[mask], mode='markers',
                                    marker=dict(size=atom_size, color=color), name=str(species)))

    # Plot lattice vectors
    if plot_lattice:
        lattice_vectors = np.array(structure.lattice.matrix)
        fig.add_trace(go.Scatter3d(x=[0, lattice_vectors[0, 0]], y=[0, lattice_vectors[0, 1]], z=[0, lattice_vectors[0, 2]],
                                    mode='lines', line=dict(color='red'), name='a'))
        fig.add_trace(go.Scatter3d(x=[0, lattice_vectors[1, 0]], y=[0, lattice_vectors[1, 1]], z=[0, lattice_vectors[1, 2]],
                                    mode='lines', line=dict(color='green'), name='b'))
        fig.add_trace(go.Scatter3d(x=[0, lattice_vectors[2, 0]], y=[0, lattice_vectors[2, 1]], z=[0, lattice_vectors[2, 2]],
                                    mode='lines', line=dict(color='blue'), name='c'))

    # Plot Miller indices vector
    if miller_indices_vector is not None:
        vec = structure.lattice.matrix[0]*miller_indices_vector[0]+ structure.lattice.matrix[1]*miller_indices_vector[1]+ structure.lattice.matrix[2]*miller_indices_vector[2]
        fig.add_trace(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                                    mode='lines', line=dict(color='purple'), name=f'{miller_indices_vector}'))
    if coord_axes is not None:
        coord_vectors = np.array(coord_axes)
        fig.add_trace(go.Scatter3d(x=[0, coord_vectors[0, 0]], y=[0, coord_vectors[0, 1]], z=[0, coord_vectors[0, 2]],
                                    mode='lines', line=dict(color='black'), name='x'))
        fig.add_trace(go.Scatter3d(x=[0, coord_vectors[1, 0]], y=[0, coord_vectors[1, 1]], z=[0, coord_vectors[1, 2]],
                                    mode='lines', line=dict(color='black'), name='y'))
        fig.add_trace(go.Scatter3d(x=[0, coord_vectors[2, 0]], y=[0, coord_vectors[2, 1]], z=[0, coord_vectors[2, 2]],
                                    mode='lines', line=dict(color='black'), name='z'))
        
    # Set layout
    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                        width=800,
                        margin=dict(r=20, b=10, l=10, t=10),
                        legend=dict(title='Species', orientation='h', yanchor='top', y=1.05, xanchor='right', x=1))
    fig.update_layout(scene=dict(aspectmode='data'))

    # Show plot
    fig.show()
    return fig

def plot_coords(coords, atom_types, atom_size=15,fig=None):
    # Mapping atom types to colors for visualization
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    unique_types = sorted(list(set(atom_types)))
    color_map = {species: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, species in enumerate(unique_types)}
    colors = [color_map[s] for s in unique_types]
    
    type_colors = {atom_type: colors[i] for i, atom_type in enumerate(unique_types)}

    # Create traces for each atom type
    traces = []
    for atom_type in unique_types:
        trace = go.Scatter3d(
            x=[coord[0] for coord, t in zip(coords, atom_types) if t == atom_type],
            y=[coord[1] for coord, t in zip(coords, atom_types) if t == atom_type],
            z=[coord[2] for coord, t in zip(coords, atom_types) if t == atom_type],
            mode='markers',
            marker=dict(
                size=atom_size,
                color=type_colors[atom_type],
                symbol='circle'
            ),
            name=atom_type
        )
        traces.append(trace)

    # Create layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Plot
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


# %%
