# PyView - Interactive NetCDF Python Viewer

PyView is a Python-based interactive netCDF viewer designed for Jupyter notebooks. It provides a modern ncview-like experience optimized for scientific workflows on remote servers without X11 forwarding.

## Key Features

### Instant Visualization
- **Zero Manual Setup**: Ready to explore data immediately with `pyview('your_file.nc')`

### Professional Scientific Interface
- **Four Plot Types**: Geographic maps, depth profiles, time series, Hovmöller diagrams
- **Manual Colorbar Controls**: Text-based range inputs for precise visualization control
- **Smart Validation**: Only shows valid plot types based on data dimensions
- **Context-Aware Navigation**: Dimension controls adapt intelligently to plot type
- **Theme Support**: Light and dark mode themes with toggle button

### Advanced Data Analysis
- **Real-Time Statistics**: Live min/max/mean/standard deviation updates
- **Data Quality Metrics**: Valid points count and missing data detection
- **Multi-Dimensional Navigation**: Seamless navigation through time, depth, lat/lon
- **Coordinate-Aware**: Displays coordinate values with file units

### Technical Excellence
- **No External Plot Leakage**: All visualizations contained within the widget interface
- **Memory Efficient**: Optimized for large multi-GB datasets
- **Format Flexibility**: Supports netCDF, Zarr, GRIB, and any xarray-compatible format

## Installation

### Automated Setup (Recommended)

The easiest way to set up PyView is using the provided conda environment and setup script:

```bash
# Clone or download the PyView repository
git clone https://github.com/your-repo/pyview.git
cd pyview

# Run the automated setup script
bash setup_env.sh

# Activate the environment
conda activate pyview

# Start Jupyter Lab
jupyter lab
```

The setup script will:
- Create a conda environment with all required dependencies
- Install PyView-specific packages and JupyterLab extensions
- Verify the installation
- Provide clear next steps

### Manual Installation

If you prefer manual installation or want to add PyView to an existing environment:

#### Using Conda (Recommended)
```bash
# Create environment from the provided environment.yml
conda env create -f environment.yml
conda activate pyview

# Ensure JupyterLab widgets work properly
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

#### Using Pip
```bash
# Core dependencies
pip install xarray matplotlib ipywidgets numpy

# NetCDF and data format support
pip install netcdf4 h5netcdf zarr

# Jupyter Lab and widgets
pip install jupyterlab ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Optional performance packages
pip install scipy dask bottleneck
```

### Verification

Test your installation:
```python
# In Python or Jupyter
from pyview import pyview
print("PyView ready!")

# Test with sample data
viewer = pyview('your_test_file.nc')
```

### Dependencies

**Core Requirements:**
- Python >= 3.8
- xarray >= 0.20.0
- matplotlib >= 3.5.0
- ipywidgets >= 7.6.0
- numpy >= 1.20.0

**Data Format Support:**
- netcdf4 >= 1.5.0 (NetCDF files)
- h5netcdf >= 0.14.0 (HDF5-based NetCDF)
- zarr >= 2.10.0 (Zarr arrays)

**Optional but Recommended:**
- scipy (additional scientific functions)
- dask (large dataset handling)
- bottleneck (faster xarray operations)
- pandas >= 1.3.0 (data manipulation)

## Usage

### Instant Start (Recommended)
```python
from pyview import pyview

# One-line setup with automatic plot generation
viewer = pyview('/path/to/your/ocean_data.nc')
# Plot appears immediately! No manual configuration needed.
```

### Advanced Usage
```python
from pyview import PyView

# Create viewer object for more control
viewer = PyView('/path/to/your/file.nc')

# Access the underlying xarray dataset
print(f"Variables: {list(viewer.ds.data_vars)}")
print(f"Dimensions: {list(viewer.ds.dims)}")

# Show the interface when ready
viewer.show()

# Close when done
viewer.close()
```

### Dark Mode
```python
# Enable dark mode on startup
viewer = pyview('/path/to/your/file.nc', dark_mode=True)

# Or use the Dark/Light toggle button in the interface
```

## Interface Guide

### Main Components

1. **Variable and View Panel** (Left)
   - Theme toggle button (Dark/Light mode)
   - Variable selector with dimensions display
   - Plot type selector (Map/Depth/Time/Hovmöller)
   - Colormap selection
   - Automatic/Manual colorbar range controls

2. **Data Visualization Panel** (Right)
   - Real-time data statistics
   - Plot area (600×450px)
   - Coordinate-aware axes and colorbars

3. **Navigation Controls** (Bottom)
   - Previous/Next buttons with wrap-around
   - Coordinate displays with actual values
   - Context-aware dimension controls

4. **Status Area** (Bottom)
   - Current position coordinates
   - Real-time updates during navigation

### Plot Types Available

#### Geographic Maps
- **When**: Variables have lat × lon dimensions
- **Navigation**: Time slider, depth slider
- **Perfect for**: Sea surface temperature, atmospheric fields, satellite data

#### Depth Profiles
- **When**: Variables have depth dimension
- **Navigation**: Lat/lon sliders to select location, time slider
- **Perfect for**: Ocean temperature profiles, salinity stratification

#### Time Series
- **When**: Variables have time dimension
- **Navigation**: Lat/lon/depth sliders to select point
- **Perfect for**: Temporal evolution at specific locations

#### Hovmöller Diagrams
- **When**: Variables have depth + lat/lon dimensions
- **Navigation**: Time slider, one spatial dimension
- **Perfect for**: Space-time analysis, upwelling patterns

### Smart Features

#### Intelligent Plot Validation
PyView detects which plot types are possible for each variable:
- Only shows available plot types in the interface
- Prevents impossible configurations
- Guides users to valid visualizations

#### Context-Aware Navigation
Navigation controls adapt to the selected plot type:
- **Map view**: Navigate time/depth (lat/lon are plotted)
- **Depth profile**: Navigate lat/lon/time (depth is plotted)
- **Time series**: Navigate lat/lon/depth (time is plotted)
- **Hovmöller**: Navigate time + one spatial dimension

#### Professional Colorbar Controls
- **Auto Range**: Automatically uses data min/max
- **Manual Range**: Text inputs for precise control
- **Direct Input**: Type numbers directly (e.g., 15.5 or -2.3)
- **Instant Updates**: Changes apply immediately

#### Theme Support
- **Light Theme**: Clean, professional appearance with white backgrounds
- **Dark Theme**: Dark blue-gray palette optimized for low-light environments
- **Toggle Button**: Switch themes instantly without restarting
- **Consistent Styling**: All widgets properly themed


## Advanced Features

### Multiple Colormap Options
Available colormaps include viridis, plasma, coolwarm, RdBu_r, seismic, and ocean. Select from dropdown or set programmatically.

### Real-Time Statistics
- **Min/Max**: Data range for current slice
- **Mean/Std**: Statistical summary
- **Valid Points**: Data quality assessment
- **Missing Data**: NaN/fill value detection

### Navigation Shortcuts
- **Previous/Next Buttons**: Navigate dimensions with wrap-around
- **Coordinate Display**: See actual lat/lon/depth/time values
- **Middle Position**: Quick jump to middle of dimension range

## Customization


### Export Functionality
```python
# Save current plot
viewer.figure.savefig('current_plot.png', dpi=300, bbox_inches='tight')

# Export data slice
current_data = viewer.get_plot_data()[0]
current_data.to_netcdf('current_slice.nc')
```

## Troubleshooting

### Common Issues

**"No module named 'xarray'"**
```bash
pip install xarray netcdf4
```

**"Widgets not displaying"**
```bash
# Jupyter Lab
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Jupyter Notebook
jupyter nbextension enable --py widgetsnbextension
```

**"Large file performance issues"**
- Use xarray's `chunks` parameter for very large files
- Consider temporal/spatial subsets for initial exploration
- PyView is optimized but 50GB+ files may still be slow

**"Plot not updating"**
- Check that you have valid data (not all NaN)
- Verify dimension names match expected patterns
- Try manual colorbar range if auto-range fails

### Performance Optimization

1. **Memory Management**: Close viewers when done with `viewer.close()`
2. **Data Chunking**: Use xarray's dask integration for huge files
3. **Region Selection**: Load only needed spatial/temporal domains
4. **Format Choice**: HDF5-based netCDF often performs better than classic

## Integration

### Jupyter Workflows
```python
# Perfect for data exploration notebooks
viewer = pyview('model_output.nc')

# Combine with analysis
import xarray as xr
ds = xr.open_dataset('model_output.nc')
sst = ds.temp.isel(depth=0).mean('time')
sst.plot()  # Traditional static plot

viewer2 = pyview('model_output.nc')  # Interactive exploration
```

### Remote Servers
PyView works perfectly on HPC systems without X11 forwarding. There are no matplotlib backend issues as all plotting is contained within Jupyter widgets.

## Performance Tips

1. **Start with Overview**: Use auto-plot to quickly assess data structure
2. **Zoom into Regions**: Use manual colorbar to focus on specific ranges
3. **Time Series Analysis**: Switch to time series for temporal patterns
4. **Multi-Variable Comparison**: Open multiple viewers to compare variables
5. **Export Key Plots**: Save publication-ready figures with high DPI

## Best Practices

1. **Data Exploration Workflow**:
   - Start with auto-generated plot for overview
   - Use different plot types to understand data structure
   - Apply manual colorbar ranges for detailed analysis
   - Export key visualizations for presentations

2. **Scientific Analysis**:
   - Geographic maps for spatial patterns
   - Depth profiles for vertical structure
   - Time series for temporal evolution
   - Hovmöller diagrams for space-time dynamics

3. **Performance**:
   - Close viewers when done
   - Use appropriate data chunking
   - Consider regional subsets for large files

## Contributing

PyView is designed to be extensible and contribution-friendly:

- **Additional Plot Types**: 3D visualization, vector plots, multi-panel displays
- **Enhanced Statistics**: Histograms, spatial correlations, spectral analysis
- **Export Features**: Animations, data subsets, publication formats
- **Performance**: Optimization for very large datasets, parallel processing
- **Format Support**: Additional file formats, cloud storage integration

## License

MIT License - Open source and free for all scientific use.

## Author

Urs Hofmann Elizondo, 19/09/2025

## Acknowledgments

Inspired by the legendary ncview tool by David W. Pierce, reimagined for modern Jupyter-based scientific workflows in ocean modeling, atmospheric science, and climate research.

---

**Perfect for Ocean Modeling • Ideal for Climate Science • Great for Satellite Data**
