#%%
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from scipy.spatial import Voronoi,voronoi_plot_3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

#%%
# 1. Load or create multiple crystal structures (grains)
grain1 = LammpsData.from_file("/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/atom_labels/data.B11-Ce-CBC",atom_style='atomic').structure 
grain2 = LammpsData.from_file("/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/atom_labels/data.B11-Cp-CBC",atom_style='atomic').structure
grain3 = LammpsData.from_file("/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/atom_labels/data.B12-CBC",atom_style='atomic').structure
grain4 = LammpsData.from_file("/home/kimia.gh/blue2/B4C_ML_Potential/analysis_scripts/data_files/atom_labels/data.B12-CCC",atom_style='atomic').structure

#%%
# 2. Define seed points for grains in 3D space
# These seed points will serve as the centers of the grains in the Voronoi tessellation

#size of desired polycrystal
dims = [160,160,1000]

#pick 50 random seed points 
seed_points = np.random.rand(40, 3) * dims

#assign a proportion of the seed points to each grain based on relative abundance
g1_abundance = 0.35
g2_abundance = 0.22
g3_abundance = 0.25
g4_abundance = 0.18

# assign seed points to grains based on abundance
g1_points = seed_points[:int(g1_abundance*len(seed_points))]
g2_points = seed_points[int(g1_abundance*len(seed_points)):int((g1_abundance+g2_abundance)*len(seed_points))]
g3_points = seed_points[int((g1_abundance+g2_abundance)*len(seed_points)):int((g1_abundance+g2_abundance+g3_abundance)*len(seed_points))]
g4_points = seed_points[int((g1_abundance+g2_abundance+g3_abundance)*len(seed_points)):]

points = pd.DataFrame(seed_points, columns=['x','y','z'])

# assign a grain to each seed point based on the abundance
points['grain'] = np.nan
points.loc[points.index[:len(g1_points)], 'grain'] = 'B11-Cp-CBC'
points.loc[points.index[len(g1_points):len(g1_points)+len(g2_points)], 'grain'] = 'B11-Ce-CBC'
points.loc[points.index[len(g1_points)+len(g2_points):len(g1_points)+len(g2_points)+len(g3_points)], 'grain'] = 'B12-CBC'
points.loc[points.index[len(g1_points)+len(g2_points)+len(g3_points):], 'grain'] = 'B12-CCC'

points_df = points.copy()
#visualize the seed points
fig = px.scatter_3d(points, x='x', y='y', z='z', color='grain')
fig.update_layout(scene_aspectmode='data')
fig.show()
#%%

# 3. Perform Voronoi tessellation
vor = Voronoi(seed_points)

# 4. Visualize the Voronoi tessellation
# go.scatter(x=vor.vertices[:, 0], y=vor.vertices[:, 1], z=vor.vertices[:, 2], color='red') 

# 5. Assign each Voronoi region to a crystal structure (this step is conceptual)
# For each Voronoi region, you would map it to one of the grains (grain1, grain2, grain3)
# This step requires filling the Voronoi region with atoms from the crystal structure
# and ensuring the structure is periodic if needed.

# You'll have to implement the logic to place the atoms of each grain inside the corresponding Voronoi region.

# %%
import matplotlib.pyplot as plt
fig = voronoi_plot_3d(vor)
plt.show()

#%%

df = points_df.copy()

# Step 2: Perform Voronoi Tessellation using scipy.spatial.Voronoi
# Extract the 3D points from the DataFrame
points = df[["x", "y", "z"]].values

# Conduct Voronoi tessellation
vor = Voronoi(points)

# Step 3: Assign colors to each grain
# Map grain strings to integers to use for coloring
grain_labels = df["grain"].unique()
grain_color_map = {grain: i for i, grain in enumerate(grain_labels)}
colors = [grain_color_map[grain] for grain in df["grain"]]

# Step 4: Create a Plotly figure
fig = go.Figure()

# Plot the seed points (grains) with different colors
fig.add_trace(go.Scatter3d(
    x=points[:, 0], 
    y=points[:, 1], 
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=10,
        color=colors,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Grain")
    ),
    text=[f'Grain: {grain}' for grain in df["grain"]],
    name="Grain Centers"
))

# Plot Voronoi vertices
# fig.add_trace(go.Scatter3d(
#     x=vor.vertices[:, 0],
#     y=vor.vertices[:, 1],
#     z=vor.vertices[:, 2],
#     mode='markers',
#     marker=dict(
#         size=4,
#         color='black'
#     ),
#     name="Voronoi Vertices"
# ))

# Step 5: Plot the Voronoi regions (edges)
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        # Extract the vertices of the Voronoi region
        polygon = np.array([vor.vertices[i] for i in region])
        if len(polygon) > 2:
            # Plot the polygons (only plotting edges for simplicity)
            for i in range(len(polygon)):
                fig.add_trace(go.Scatter3d(
                    x=[polygon[i, 0], polygon[(i + 1) % len(polygon), 0]],
                    y=[polygon[i, 1], polygon[(i + 1) % len(polygon), 1]],
                    z=[polygon[i, 2], polygon[(i + 1) % len(polygon), 2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

# Customize plot layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=6)
    ),
    title="3D Voronoi Tessellation with Grain Assignments",
    showlegend=True
)
# limit axes to dims 
fig.update_layout(scene=dict(xaxis=dict(range=[0, dims[0]]), yaxis=dict(range=[0, dims[1]]), zaxis=dict(range=[0, dims[2]])))
# Show the plot
fig.show()

# %%
