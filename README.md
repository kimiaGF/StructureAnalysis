# Cuboidal Unit Cell Generator for Molecular Simulations

This repository contains Python scripts for generating and manipulating cuboidal unit cells from molecular structures. These scripts are particularly useful for analyzing crystal structures, performing high-performance molecular dynamics simulations, and preparing input files for LAMMPS simulations.

## Features

- **Rotation and Transformation:** Rotate molecular structures to align specific axes and transform lattice vectors.
- **Cuboidal Cell Generation:** Generate a cuboidal unit cell with specified dimensions and orientations from rhombohedral lattice vectors.
- **High-Precision Computation:** Handle high-tolerance adjustments for lattice alignment and duplication.
- **LAMMPS Integration:** Create LAMMPS data files for further simulations.
- **Visualization:** Plot structures, lattice vectors, and supercells with tools like Plotly.

## Prerequisites

The following Python libraries are required to run the scripts:

- `pymatgen`
- `numpy`
- `plotly`
- `pandas`
- `multiprocessing`
- `logging`

Install them using pip:

```bash
pip install pymatgen numpy plotly pandas
```

## Usage

### 1. Generate Cuboidal Unit Cell

To generate a cuboidal unit cell from a structure file:

```python
from cuboidal_cell_generator import generate_cuboidal_unit_cell

path_to_data = "path/to/lammps/data/file"
angle = 45  # Rotation angle in degrees
tol = 0.1  # Tolerance for alignment

cuboidal_unit_cell = generate_cuboidal_unit_cell(path_to_data, angle, tol)
```

### 2. Rotate Lattice and Find Best Unit Cell

```python
from cuboidal_cell_generator import rotate_lattice, FindUnitCell

rotated_struct, new_axes = rotate_lattice(struct, angle=30)
BestLatDim, BestLatVect, angles = FindUnitCell(rotated_struct, tol=0.09, n=51)
```

### 3. Create Supercells

The `FindUnitCell` and `generate_cuboidal_unit_cell` functions calculate optimal lattice vectors for creating supercells, which can then be used for molecular simulations.

### 4. Save Output for LAMMPS

LAMMPS-compatible data files can be generated and saved:

```python
from pymatgen.io.lammps.data import LammpsData

lammps_data = LammpsData.from_structure(cuboidal_unit_cell, atom_style="atomic")
lammps_data.write_file("output/data.file")
```

## Logging

The script logs operations and errors to `logs/transformation.log`. Adjust logging levels in the script if needed.

## Example

The following example generates cuboidal unit cells for multiple orientations of boron carbide (B4C):

```python
fs = glob('data_files/*.data')
thetas = [0, 30, 45, 60, 90]

for theta in thetas:
    for f in fs:
        unit_cell = generate_cuboidal_unit_cell(f, angle=theta, tol=0.1)
        print(unit_cell.lattice)
```

## File Structure

- `cuboidal_cell_generator.py`: Main script for generating and manipulating cuboidal unit cells.
- `logs/`: Directory containing log files.
- `data_files/`: Directory for input and output structure files.

## Author
Kimia Ghaffari

## License
MIT License

## Acknowledgments
Special thanks to the developers of [pymatgen](https://pymatgen.org) and NVIDIA for their support in enabling high-performance computing solutions at the University of Florida.
