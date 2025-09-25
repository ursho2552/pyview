"""
PyView - NetCDF viewer for Jupyter notebooks
====================================================

The data can be displayed as:
- Hovmöller plots (depth vs lat with variable longitude)
- Previous/Next navigation buttons with wrap-around
- Compact navigation area matching upper panel widths

Author: Urs Hofmann Elizondo
Date: 19/09/2025
"""

import xarray as xr
import matplotlib.figure as mfig
from matplotlib.backends.backend_agg import FigureCanvasAgg
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import io

# Common dimension keywords for flexible matching
DIMENSION_KEYWORDS = {
    'lat': ['lat', 'yt', 'yu', 'y', 'latitude'],
    'lon': ['lon', 'xt', 'xu', 'x', 'longitude'],
    'depth': ['dep', 'lev', 'z', 'st', 'sw', 'depth', 'level'],
    'time': ['tim', 'time']
}

class PyView:
    """PyView class"""

    def __init__(self, filepath, dark_mode=False, dim_mapping=None):
        self.filepath = filepath
        self.ds = xr.open_mfdataset(filepath, decode_times=False, data_vars='all')
        self.current_var = None
        self.dark_mode = dark_mode

        # Get all data variables
        self.vars = list(self.ds.data_vars)
        self.standard_mapping = dim_mapping
        self.dimension_names = self.get_dimension_names(dim_mapping=dim_mapping)

        # Storage for widgets
        self.sliders = {}
        self.coord_displays = {}

        # Apply theme styling
        self.apply_theme()

        # Create the interface
        self.setup_interface()

    def get_dimension_names(self, dim_mapping=None):
        """Get standardized dimension names based on provided mapping or common keywords"""
        if dim_mapping is None:
            dim_mapping = {}

        # Default dimension names
        dim_names = {
            'lat': None,
            'lon': None,
            'depth': None,
            'time': None
        }

        # Check for user-provided mappings first
        dims_list = self.ds[self.current_var].dims if self.current_var else self.ds.dims
        for key in dim_names.keys():
            if key in dim_mapping and dim_mapping[key] in dims_list:
                dim_names[key] = dim_mapping[key]

        # Auto-detect dimensions from all dataset dimensions (not just current variable)
        for dim in dims_list:
            dim_lower = dim.lower()

            # Check each dimension type
            for dim_type, keywords in DIMENSION_KEYWORDS.items():
                if dim_names[dim_type] is None:  # Only assign if not already found
                    if any(keyword in dim_lower for keyword in keywords) or dim in keywords:
                        dim_names[dim_type] = dim
                        break

        return dim_names

    def has_dimension_type(self, var_name, dim_type):
        """Check if a variable has a specific dimension type"""
        if not var_name or dim_type not in self.dimension_names:
            return False

        dim_name = self.dimension_names[dim_type]
        if not dim_name:
            return False

        var = self.ds[var_name]
        return dim_name in var.dims

    def get_dimension_name(self, dim_type):
        """Get the actual dimension name for a dimension type"""
        return self.dimension_names.get(dim_type)

    def show_dimension_mapping(self):
        """Display current dimension mapping"""
        print("Current Dimension Mapping:")
        print("=" * 30)
        for dim_type, dim_name in self.dimension_names.items():
            print(f"{dim_type.capitalize()}: {dim_name or 'Not found'}")

        print(f"\nDataset dimensions: {list(self.ds.dims.keys())}")

    def apply_theme(self):
        """Apply dark or light theme styling"""
        from IPython.display import HTML, display as ipy_display

        if self.dark_mode:
            # Dark theme CSS
            theme_css = """
            <style>
            /* Dark theme for PyView */
            .widget-html, .widget-html-content {
                background-color: #2d3748 !important;
                color: #e2e8f0 !important;
            }

            .widget-dropdown select, .widget-text input {
                background-color: #4a5568 !important;
                color: #e2e8f0 !important;
                border: 1px solid #718096 !important;
            }

            .widget-dropdown select:focus, .widget-text input:focus {
                border-color: #63b3ed !important;
                box-shadow: 0 0 0 1px #63b3ed !important;
            }

            .widget-button {
                background-color: #4a5568 !important;
                color: #e2e8f0 !important;
                border: 1px solid #718096 !important;
            }

            .widget-button:hover {
                background-color: #2d3748 !important;
                border-color: #63b3ed !important;
            }

            .widget-button.mod-info {
                background-color: #3182ce !important;
                border-color: #3182ce !important;
            }

            .widget-button.mod-info:hover {
                background-color: #2c5282 !important;
            }

            .widget-radio-box label {
                color: #e2e8f0 !important;
            }

            .widget-checkbox input[type="checkbox"] {
                background-color: #4a5568 !important;
                border: 1px solid #718096 !important;
            }

            .widget-label {
                color: #e2e8f0 !important;
            }

            .widget-readout {
                color: #e2e8f0 !important;
                background-color: #4a5568 !important;
            }

            /* Container backgrounds */
            .widget-vbox, .widget-hbox {
                background-color: #1a202c !important;
            }

            /* Image widget border */
            .widget-image {
                border: 1px solid #4a5568 !important;
            }

            /* HTML content styling */
            .widget-html-content h4 {
                color: #63b3ed !important;
                margin: 5px 0 !important;
            }

            .widget-html-content hr {
                border-color: #4a5568 !important;
            }

            .widget-html-content small {
                color: #cbd5e0 !important;
            }

            .widget-html-content b {
                color: #e2e8f0 !important;
            }
            </style>
            """
        else:
            # Light theme CSS (default/clean)
            theme_css = """
            <style>
            /* Light theme for PyView */
            .widget-html, .widget-html-content {
                background-color: #ffffff !important;
                color: #2d3748 !important;
            }

            .widget-dropdown select, .widget-text input {
                background-color: #ffffff !important;
                color: #2d3748 !important;
                border: 1px solid #cbd5e0 !important;
            }

            .widget-dropdown select:focus, .widget-text input:focus {
                border-color: #3182ce !important;
                box-shadow: 0 0 0 1px #3182ce !important;
            }

            .widget-button {
                background-color: #f7fafc !important;
                color: #2d3748 !important;
                border: 1px solid #cbd5e0 !important;
            }

            .widget-button:hover {
                background-color: #edf2f7 !important;
                border-color: #3182ce !important;
            }

            .widget-button.mod-info {
                background-color: #3182ce !important;
                color: #ffffff !important;
                border-color: #3182ce !important;
            }

            .widget-button.mod-info:hover {
                background-color: #2c5282 !important;
            }

            .widget-radio-box label {
                color: #2d3748 !important;
            }

            .widget-checkbox input[type="checkbox"] {
                background-color: #ffffff !important;
                border: 1px solid #cbd5e0 !important;
            }

            .widget-label {
                color: #2d3748 !important;
            }

            .widget-readout {
                color: #2d3748 !important;
                background-color: #f7fafc !important;
            }

            /* Container backgrounds */
            .widget-vbox, .widget-hbox {
                background-color: #ffffff !important;
            }

            /* Image widget border */
            .widget-image {
                border: 1px solid #cbd5e0 !important;
            }

            /* HTML content styling */
            .widget-html-content h4 {
                color: #2d3748 !important;
                margin: 5px 0 !important;
            }

            .widget-html-content hr {
                border-color: #e2e8f0 !important;
            }

            .widget-html-content small {
                color: #4a5568 !important;
            }

            .widget-html-content b {
                color: #2d3748 !important;
            }
            </style>
            """

        # Apply the CSS
        ipy_display(HTML(theme_css))

    def validate_plot_types_for_variable(self, var_name):
        """Validate which plot types are available for a given variable"""
        if not var_name:
            return {'map': False, 'depth': False, 'time': False, 'hovmoller': False}

        # Use the centralized dimension detection
        has_lat = self.has_dimension_type(var_name, 'lat')
        has_lon = self.has_dimension_type(var_name, 'lon')
        has_depth = self.has_dimension_type(var_name, 'depth')
        has_time = self.has_dimension_type(var_name, 'time')

        # Count spatial/temporal dimensions
        spatial_temporal_count = sum([has_lat, has_lon, has_depth, has_time])

        validation = {
            'map': has_lat and has_lon and spatial_temporal_count >= 2,
            'depth': has_depth and spatial_temporal_count >= 1,
            'time': has_time and spatial_temporal_count >= 1,
            'hovmoller': has_depth and (has_lat or has_lon) and spatial_temporal_count >= 2
        }

        return validation

    def get_navigable_dimensions_for_plot_type(self, var_name, plot_type):
        """Get which dimensions should be navigable for a given plot type"""
        if not var_name:
            return set()

        var = self.ds[var_name]
        dims = set(var.dims)
        navigable = set()

        # Get dimension names
        lat_dim = self.get_dimension_name('lat')
        lon_dim = self.get_dimension_name('lon')
        depth_dim = self.get_dimension_name('depth')
        time_dim = self.get_dimension_name('time')

        if plot_type == 'map':
            # Can navigate: depth, time (but NOT lat/lon which are plot axes)
            for dim in dims:
                if dim == depth_dim or dim == time_dim:
                    navigable.add(dim)

        elif plot_type == 'depth':
            # Can navigate: lat, lon, time (but NOT depth which is plot axis)
            for dim in dims:
                if dim == lat_dim or dim == lon_dim or dim == time_dim:
                    navigable.add(dim)

        elif plot_type == 'time':
            # Can navigate: lat, lon, depth (but NOT time which is plot axis)
            for dim in dims:
                if dim == lat_dim or dim == lon_dim or dim == depth_dim:
                    navigable.add(dim)

        elif plot_type == 'hovmoller':
            # Can navigate: time and the NON-PLOTTED spatial dimension
            # (depth and one spatial dimension are plot axes)
            for dim in dims:
                if dim == time_dim:
                    navigable.add(dim)
                # For Hovmöller, we plot depth vs lat by default, so lon is navigable
                elif dim == lon_dim:
                    navigable.add(dim)

        return navigable

    def setup_interface(self):
        """Setup the ncview-style interface"""

        # Add theme toggle button at the top
        self.theme_toggle = widgets.Button(
            description='Dark' if not self.dark_mode else 'Light',
            button_style='info',
            layout=widgets.Layout(width='100px', margin='0 0 10px 0')
        )
        self.theme_toggle.on_click(self.toggle_theme)

        # === LEFT PANEL: Variable Selection and Plot Controls ===

        # Variable selector
        self.var_selector = widgets.Dropdown(
            options=[(f"{var}", var) for var in self.vars],
            description='Variable:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='290px')
        )
        self.var_selector.observe(self.on_var_change, names='value')


        # Plot type selector (like ncview's view menu)
        self.plot_type = widgets.RadioButtons(
            options=[('Geographic Map', 'map'), ('Depth Profile', 'depth'), ('Time Series', 'time'), ('Hovmöller', 'hovmoller')],
            value='map',
            description='View Type:',
            style={'description_width': '80px'}
        )
        self.plot_type.observe(self.on_plot_type_change, names='value')

        # Colormap selector
        self.cmap_selector = widgets.Dropdown(
            options=['viridis', 'plasma', 'coolwarm', 'RdBu_r', 'seismic', 'ocean'],
            value='viridis',
            description='Colormap:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='290px')
        )
        self.cmap_selector.observe(self.update_plot, names='value')

        # Colorbar range controls
        self.auto_range_checkbox = widgets.Checkbox(
            value=True,
            description='Auto range',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='290px')
        )
        self.auto_range_checkbox.observe(self.on_range_mode_change, names='value')

        # Manual range inputs (initially disabled)
        self.vmin_input = widgets.Text(
            value='0.0',
            description='Min:',
            style={'description_width': '40px'},
            layout=widgets.Layout(width='140px'),
            disabled=True
        )
        self.vmin_input.observe(self.on_range_input_change, names='value')

        self.vmax_input = widgets.Text(
            value='1.0',
            description='Max:',
            style={'description_width': '40px'},
            layout=widgets.Layout(width='140px'),
            disabled=True
        )
        self.vmax_input.observe(self.on_range_input_change, names='value')

        # Range controls in horizontal layout
        self.range_controls = widgets.HBox([
            self.vmin_input,
            self.vmax_input
        ], layout=widgets.Layout(width='290px'))

        # Dimension controls container
        self.dim_controls = widgets.VBox()

        # Left panel assembly
        left_panel = widgets.VBox([
            self.theme_toggle,
            widgets.HTML("<h4>Variable & View</h4>"),
            self.var_selector,
            self.plot_type,
            self.cmap_selector,
            self.auto_range_checkbox,
            self.range_controls,
            widgets.HTML("<hr>"),
            widgets.HTML("<h4>Dimensions</h4>"),
            self.dim_controls
        ], layout=widgets.Layout(width='320px', padding='10px'))

        # === RIGHT PANEL: Plot Display ===

        # Statistics display (above plot for tighter layout)
        self.stats_display = widgets.HTML(
            value="<b>Statistics:</b> --",
            layout=widgets.Layout(width='600px', margin='0 0 5px 0')
        )

        # Plot widget
        self.plot_widget = widgets.Image(
            format='png',
            width=600,
            height=450,
            layout=widgets.Layout(border='1px solid #ccc')
        )

        right_panel = widgets.VBox([
            widgets.HTML("<h4>Data Visualization</h4>"),
            self.stats_display,
            self.plot_widget
        ], layout=widgets.Layout(padding='10px'))

        # === BOTTOM PANEL: Navigation Controls ===

        # Navigation controls container - match width of left Variable&View panel
        self.nav_controls = widgets.HBox(
            layout=widgets.Layout(
                justify_content='flex-start',
                width='720px',  # Match width of left Variable&View panel...
                margin='0'
            )
        )

        # === STATUS AREA: Coordinates only (statistics moved above plot) ===

        # Current coordinates display
        self.coord_status = widgets.HTML(
            value="<b>Current Position:</b> Select a variable to begin",
            layout=widgets.Layout(width='800px')
        )

        status_area = widgets.HBox([
            self.coord_status
        ], layout=widgets.Layout(justify_content='flex-start', padding='5px'))

        # === MAIN LAYOUT ===

        # Top section: left and right panels side by side
        main_section = widgets.HBox([
            left_panel,
            right_panel
        ], layout=widgets.Layout(align_items='flex-start'))

        # Complete interface
        self.interface = widgets.VBox([
            main_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h4>Navigation</h4>"),
            self.nav_controls,
            widgets.HTML("<hr>"),
            status_area
        ])

    def toggle_theme(self, button):
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode

        # Update button text
        if self.dark_mode:
            button.description = 'Light'
        else:
            button.description = 'Dark'

        # Reapply theme
        self.apply_theme()

    def create_dimension_controls(self, var_name):
        """Create dimension controls for a variable"""
        var = self.ds[var_name]
        controls = []
        nav_controls = []

        # Preserve existing dimension indices where possible
        old_dim_indices = self.dim_indices.copy() if hasattr(self, 'dim_indices') else {}
        self.dim_indices = {}  # Store current indices for each dimension
        self.coord_displays = {}
        self.nav_buttons = {}

        # Get current plot type and determine navigable dimensions
        current_plot_type = self.plot_type.value
        navigable_dims = self.get_navigable_dimensions_for_plot_type(var_name, current_plot_type)

        for dim in var.dims:
            size = var.sizes[dim]

            if size > 1:
                # Initialize index - preserve old index if dimension exists and index is valid
                if dim in old_dim_indices and old_dim_indices[dim] < size:
                    self.dim_indices[dim] = old_dim_indices[dim]
                else:
                    self.dim_indices[dim] = 0

                # Check if this dimension should be navigable for current plot type
                is_navigable = dim in navigable_dims

                # Dimension info in left panel
                if is_navigable:
                    dim_info = widgets.HTML(f"<b>{dim}</b>: {size} points")
                else:
                    dim_info = widgets.HTML(f"<b>{dim}</b>: {size} points <i>(fixed for {current_plot_type} view)</i>")
                controls.append(dim_info)

                # Navigation buttons for bottom panel (only for navigable dimensions)
                if is_navigable:
                    prev_btn = widgets.Button(
                        description='◀',
                        layout=widgets.Layout(width='40px', height='30px'),
                        button_style='info'
                    )
                    next_btn = widgets.Button(
                        description='▶',
                        layout=widgets.Layout(width='40px', height='30px'),
                        button_style='info'
                    )

                    # Connect button events with wrap-around logic
                    prev_btn.on_click(lambda b, d=dim: self.navigate_dimension(d, -1))
                    next_btn.on_click(lambda b, d=dim: self.navigate_dimension(d, 1))

                    self.nav_buttons[dim] = {'prev': prev_btn, 'next': next_btn}

                    # Current index display
                    current_idx = self.dim_indices[dim]
                    index_display = widgets.HTML(
                        value=f"<small><b>{current_idx + 1}/{size}</b></small>",
                        layout=widgets.Layout(width='50px', text_align='center')
                    )

                    # Coordinate input field
                    coord_input = widgets.Text(
                        placeholder="0.0",
                        layout=widgets.Layout(width='80px', height='25px'),
                        style={'description_width': '0px'}
                    )
                    coord_input.on_submit(lambda widget, d=dim: self.on_coord_input_submit(widget, d))
                    self.coord_displays[dim] = coord_input
                    self.update_coord_display(dim, current_idx)

                    # Compact navigation group
                    nav_group = widgets.VBox([
                        widgets.HTML(f"<small><b>{dim}</b></small>"),
                        widgets.HBox([prev_btn, index_display, next_btn],
                                    layout=widgets.Layout(justify_content='center')),
                        coord_input
                    ], layout=widgets.Layout(align_items='center', margin='0 5px', width='250px', height='120px', padding='1px'))

                    nav_controls.append(nav_group)
                else:
                    # For non-navigable dimensions, still set up coord display but no input interaction
                    current_idx = self.dim_indices[dim]
                    coord_input = widgets.Text(
                        placeholder="0.0",
                        layout=widgets.Layout(width='80px', height='25px'),
                        disabled=True,  # Non-navigable dimensions are read-only
                        style={'description_width': '0px'}
                    )
                    self.coord_displays[dim] = coord_input
                    self.update_coord_display(dim, current_idx)

        # Update left panel dimension info
        self.dim_controls.children = controls

        # Update bottom navigation panel (only navigable dimensions)
        self.nav_controls.children = nav_controls

    def on_range_mode_change(self, change):
        """Handle auto/manual range mode change"""
        auto_mode = change['new']

        # Enable/disable manual range inputs
        self.vmin_input.disabled = auto_mode
        self.vmax_input.disabled = auto_mode

        if not auto_mode and self.current_var:
            # Switch to manual mode - set current data range as starting values
            self.update_range_inputs_from_data()

        # Update plot
        self.update_plot()

    def on_range_input_change(self, change):
        """Handle manual range input changes (text to float conversion)"""
        # Only update plot if we're in manual mode
        if not self.auto_range_checkbox.value:
            self.update_plot()

    def get_manual_range(self):
        """Parse manual range inputs and return as floats"""
        try:
            vmin = float(self.vmin_input.value)
            vmax = float(self.vmax_input.value)
            # Ensure valid range
            if vmin >= vmax:
                return None, None
            return vmin, vmax
        except (ValueError, TypeError):
            # Invalid input, return None to use auto range
            return None, None

    def update_range_inputs_from_data(self):
        """Update range inputs with current data range"""
        if not self.current_var:
            return

        try:
            data_slice, plot_dims, slice_dict = self.get_plot_data()
            if data_slice is not None:
                values = data_slice.values
                flat_vals = values.flatten()
                valid_vals = flat_vals[~np.isnan(flat_vals)]
                if len(valid_vals) > 0:
                    self.vmin_input.value = str(float(np.min(valid_vals)))
                    self.vmax_input.value = str(float(np.max(valid_vals)))
        except:
            pass

    def navigate_dimension(self, dim, direction):
        """Navigate dimension with wrap-around"""
        if dim not in self.dim_indices:
            return

        current_idx = self.dim_indices[dim]
        max_idx = self.ds.sizes[dim] - 1

        # Calculate new index with wrap-around
        new_idx = current_idx + direction
        if new_idx > max_idx:
            new_idx = 0  # Wrap to beginning
        elif new_idx < 0:
            new_idx = max_idx  # Wrap to end

        self.dim_indices[dim] = new_idx

        # Update displays
        self.update_index_display(dim)
        self.update_coord_display(dim, new_idx)
        self.update_plot()

    def update_index_display(self, dim):
        """Update the index display (1/N format)"""
        current_idx = self.dim_indices[dim]
        total_size = self.ds.sizes[dim]

        # Find the index display widget and update it
        for nav_group in self.nav_controls.children:
            if isinstance(nav_group, widgets.VBox):
                title_widget = nav_group.children[0]
                if isinstance(title_widget, widgets.HTML) and dim in title_widget.value:
                    button_row = nav_group.children[1]
                    if isinstance(button_row, widgets.HBox) and len(button_row.children) >= 3:
                        index_widget = button_row.children[1]
                        if isinstance(index_widget, widgets.HTML):
                            index_widget.value = f"<small><b>{current_idx + 1}/{total_size}</b></small>"
                    break

    def update_coord_display(self, dim, index):
        """Update coordinate input field"""
        try:
            if dim in self.ds.coords:
                coord_val = self.ds.coords[dim].values[index]
                if isinstance(coord_val, (int, float, np.number)):
                    self.coord_displays[dim].value = f"{coord_val:.2f}"
                else:
                    self.coord_displays[dim].value = str(coord_val)
            else:
                self.coord_displays[dim].value = str(index)
        except:
            self.coord_displays[dim].value = str(index)

    def on_coord_input_submit(self, widget, dim):
        """Handle coordinate input field submission (Enter key)"""
        try:
            user_input = widget.value.strip()
            if not user_input:
                return

            if dim in self.ds.coords:
                # Try to find the closest coordinate value
                coord_values = self.ds.coords[dim].values

                try:
                    target_value = float(user_input)
                    # Find closest index
                    if isinstance(coord_values[0], (int, float, np.number)):
                        distances = np.abs(coord_values - target_value)
                        closest_index = int(np.argmin(distances))
                    else:
                        # Non-numeric coordinates, fallback to index
                        closest_index = int(target_value) if target_value >= 0 else 0
                except (ValueError, TypeError):
                    # If not numeric, try to find exact match
                    try:
                        closest_index = list(coord_values).index(user_input)
                    except ValueError:
                        return  # Invalid input, ignore
            else:
                # Dimension without coordinates, treat as index
                try:
                    closest_index = int(float(user_input))
                except (ValueError, TypeError):
                    return  # Invalid input, ignore

            # Validate index bounds
            dim_size = self.ds.dims[dim]
            closest_index = max(0, min(closest_index, dim_size - 1))

            # Update the dimension index and UI
            if dim in self.dim_indices:
                self.dim_indices[dim] = closest_index
                self.update_index_display(dim)
                self.update_coord_display(dim, closest_index)
                self.update_plot()

        except Exception:
            # On error, reset to current valid value
            if dim in self.dim_indices:
                self.update_coord_display(dim, self.dim_indices[dim])

    def on_plot_type_change(self, change):
        """Handle plot type change"""
        plot_type = change['new']
        self.dimension_names = self.get_dimension_names(self.standard_mapping)

        # Skip disabled options
        if plot_type.endswith('_disabled'):
            return

        if not self.current_var:
            return

        # Recreate dimension controls with new navigation validation
        self.create_dimension_controls(self.current_var)

        var = self.ds[self.current_var]
        dims = list(var.dims)

        # Get dimension names using centralized approach
        depth_dim = self.get_dimension_name('depth')
        time_dim = self.get_dimension_name('time')
        lat_dim = self.get_dimension_name('lat')
        lon_dim = self.get_dimension_name('lon')

        # Auto-configure navigation based on plot type
        if plot_type == 'map' and len(dims) >= 2:
            # For maps, set to surface/first level
            for dim in dims:
                if dim == depth_dim:
                    if dim in self.dim_indices:
                        self.dim_indices[dim] = 0  # Surface
                        self.update_index_display(dim)
                        self.update_coord_display(dim, 0)
                elif dim == time_dim:
                    if dim in self.dim_indices:
                        self.dim_indices[dim] = 0  # First time
                        self.update_index_display(dim)
                        self.update_coord_display(dim, 0)

        elif plot_type == 'depth':
            # For depth profiles, set to specific location
            for dim in dims:
                if dim == lat_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)
                elif dim == lon_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)

        elif plot_type == 'time':
            # For time series, set to specific location and depth
            for dim in dims:
                if dim == lat_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)
                elif dim == lon_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)
                elif dim == depth_dim:
                    if dim in self.dim_indices:
                        self.dim_indices[dim] = 0  # Surface level
                        self.update_index_display(dim)
                        self.update_coord_display(dim, 0)

        elif plot_type == 'hovmoller':
            # For Hovmöller, set time/other dims to first
            for dim in dims:
                if dim == time_dim:
                    if dim in self.dim_indices:
                        self.dim_indices[dim] = 0
                        self.update_index_display(dim)
                        self.update_coord_display(dim, 0)

        self.update_plot()

    def get_plot_data(self):
        """Get data for plotting based on current view type"""
        if not self.current_var:
            return None, None, {}

        var = self.ds[self.current_var]

        plot_type = self.plot_type.value

        # Skip disabled plot types
        if plot_type.endswith('_disabled'):
            return None, None, {}

        # Get current navigation positions
        slice_dict = {}
        for dim in self.dim_indices:
            slice_dict[dim] = self.dim_indices[dim]

        if plot_type == 'map':
            # Geographic map: lat x lon
            plot_dims = []
            lat_dim = self.get_dimension_name('lat')
            lon_dim = self.get_dimension_name('lon')

            for dim in var.dims:
                if dim == lat_dim:
                    plot_dims.append(dim)
                elif dim == lon_dim:
                    plot_dims.append(dim)

            if len(plot_dims) == 2:
                # Remove plot dimensions from slice
                slice_dict_copy = slice_dict.copy()
                for dim in plot_dims:
                    slice_dict_copy.pop(dim, None)

                data_slice = var.isel(slice_dict_copy)
                return data_slice, plot_dims, slice_dict_copy

        elif plot_type == 'depth':
            # Depth profile: depth vs variable value
            depth_dim = self.get_dimension_name('depth')

            if depth_dim and depth_dim in var.dims:
                # Remove depth dimension from slice (it's the plot axis)
                slice_dict_copy = slice_dict.copy()
                slice_dict_copy.pop(depth_dim, None)

                data_slice = var.isel(slice_dict_copy)
                return data_slice, [depth_dim], slice_dict_copy

        elif plot_type == 'time':
            # Time series: time vs variable value
            time_dim = self.get_dimension_name('time')

            if time_dim and time_dim in var.dims:
                # Remove time dimension from slice (it's the plot axis)
                slice_dict_copy = slice_dict.copy()
                slice_dict_copy.pop(time_dim, None)

                data_slice = var.isel(slice_dict_copy)
                return data_slice, [time_dim], slice_dict_copy

        elif plot_type == 'hovmoller':
            # Hovmöller: depth vs spatial dimension
            depth_dim = self.get_dimension_name('depth')
            lat_dim = self.get_dimension_name('lat')
            lon_dim = self.get_dimension_name('lon')

            spatial_dim = None

            # Find spatial dimension to plot (prefer lat)
            if lat_dim and lat_dim in var.dims:
                spatial_dim = lat_dim
            elif lon_dim and lon_dim in var.dims:
                spatial_dim = lon_dim

            if depth_dim and spatial_dim and depth_dim in var.dims:
                # Remove depth and spatial dimensions from slice (they're the plot axes)
                slice_dict_copy = slice_dict.copy()
                slice_dict_copy.pop(depth_dim, None)
                slice_dict_copy.pop(spatial_dim, None)

                data_slice = var.isel(slice_dict_copy)
                return data_slice, [depth_dim, spatial_dim], slice_dict_copy

        return None, None, {}

    def create_plot(self):
        """Create plot using matplotlib"""
        try:
            data_slice, plot_dims, slice_dict = self.get_plot_data()

            if data_slice is None:
                # Create an informational plot for disabled/unavailable views
                fig = mfig.Figure(figsize=(8, 6))
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)

                plot_type = self.plot_type.value
                if plot_type.endswith('_disabled'):
                    clean_plot_type = plot_type.replace('_disabled', '')
                    ax.text(0.5, 0.5, f'"{clean_plot_type.title()}" view not available\nfor variable "{self.current_var}"\n\nRequired dimensions missing',
                           ha='center', va='center', transform=ax.transAxes, fontsize=14,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, f'No data available for this view',
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

                fig.tight_layout()

                # Render to bytes
                buf = io.BytesIO()
                canvas.print_figure(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                image_bytes = buf.getvalue()
                buf.close()

                # Cleanup
                del canvas, ax, fig

                return image_bytes

            # Create figure
            fig = mfig.Figure(figsize=(8, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            values = data_slice.values
            plot_type = self.plot_type.value

            # Determine colorbar range
            vmin, vmax = None, None
            if not self.auto_range_checkbox.value:
                # Use manual range
                vmin, vmax = self.get_manual_range()

            if plot_type == 'map' and len(plot_dims) == 2:
                # Geographic map
                lat_dim, lon_dim = plot_dims
                lats = self.ds.coords[lat_dim].values
                lons = self.ds.coords[lon_dim].values

                im = ax.pcolormesh(lons, lats, values, cmap=self.cmap_selector.value,
                                 shading='auto', vmin=vmin, vmax=vmax)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'{self.current_var} - Geographic Map')

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(self.current_var)

            elif plot_type == 'depth' and len(plot_dims) == 1:
                # Depth profile
                depth_dim = plot_dims[0]
                depths = self.ds.coords[depth_dim].values

                ax.plot(values, depths, 'b-', linewidth=2)
                ax.set_xlabel(self.current_var)
                ax.set_ylabel('Depth')
                ax.set_title(f'{self.current_var} - Depth Profile')
                ax.invert_yaxis()  # Depth increases downward
                ax.grid(True, alpha=0.3)

            elif plot_type == 'time' and len(plot_dims) == 1:
                # Time series
                time_dim = plot_dims[0]
                time_coords = self.ds.coords[time_dim].values

                ax.plot(time_coords, values, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Time')
                ax.set_ylabel(self.current_var)
                ax.set_title(f'{self.current_var} - Time Series')
                ax.grid(True, alpha=0.3)

                # Rotate x-axis labels for better readability
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')

            elif plot_type == 'hovmoller' and len(plot_dims) == 2:
                # Hovmöller diagram: depth vs spatial dimension
                depth_dim, spatial_dim = plot_dims
                depths = self.ds.coords[depth_dim].values
                spatial_coords = self.ds.coords[spatial_dim].values

                # Create meshgrid for pcolormesh
                spatial_mesh, depth_mesh = np.meshgrid(spatial_coords, depths)

                im = ax.pcolormesh(spatial_mesh, depth_mesh, values,
                                 cmap=self.cmap_selector.value, shading='auto',
                                 vmin=vmin, vmax=vmax)

                # Set labels based on spatial dimension
                if 'lat' in spatial_dim.lower():
                    ax.set_xlabel('Latitude')
                    title_spatial = 'Latitude'
                elif 'lon' in spatial_dim.lower():
                    ax.set_xlabel('Longitude')
                    title_spatial = 'Longitude'
                else:
                    ax.set_xlabel(spatial_dim)
                    title_spatial = spatial_dim

                ax.set_ylabel('Depth')
                ax.set_title(f'{self.current_var} - Hovmöller ({title_spatial} vs Depth)')
                ax.invert_yaxis()  # Depth increases downward

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(self.current_var)

            else:
                ax.text(0.5, 0.5, f'View type "{plot_type}" not implemented\nfor this data structure',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)

            # Add slice information to title
            if slice_dict:
                slice_info = []
                for dim, idx in slice_dict.items():
                    if dim in self.ds.coords:
                        coord_val = self.ds.coords[dim].values[idx]
                        if isinstance(coord_val, (int, float, np.number)):
                            slice_info.append(f"{dim}={coord_val:.2f}")
                        else:
                            slice_info.append(f"{dim}={coord_val}")
                    else:
                        slice_info.append(f"{dim}[{idx}]")

                if slice_info:
                    current_title = ax.get_title()
                    ax.set_title(f"{current_title}\n{', '.join(slice_info)}", fontsize=10)

            fig.tight_layout()

            # Render to bytes
            buf = io.BytesIO()
            canvas.print_figure(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.getvalue()
            buf.close()

            # Update statistics
            self.update_stats(values)

            # Update coordinate status
            self.update_coord_status(slice_dict)

            # Cleanup
            del canvas, ax, fig

            return image_bytes

        except Exception as e:
            print(f"Plot error: {e}")
            return None

    def update_stats(self, values):
        """Update statistics display"""
        if values.size > 0:
            flat_vals = values.flatten()
            valid_vals = flat_vals[~np.isnan(flat_vals)]
            if len(valid_vals) > 0:
                stats_text = f"""<b>Statistics:</b>
                Min: {np.min(valid_vals):.3g} |
                Max: {np.max(valid_vals):.3g} |
                Mean: {np.mean(valid_vals):.3g} |
                Valid: {len(valid_vals)}/{len(flat_vals)}"""
                self.stats_display.value = stats_text
            else:
                self.stats_display.value = "<b>Statistics:</b> No valid data"
        else:
            self.stats_display.value = "<b>Statistics:</b> No data"

    def update_coord_status(self, slice_dict):
        """Update current coordinate status"""
        if slice_dict:
            coord_info = []
            for dim, idx in slice_dict.items():
                if dim in self.ds.coords:
                    coord_val = self.ds.coords[dim].values[idx]
                    if isinstance(coord_val, (int, float, np.number)):
                        coord_info.append(f"{dim}={coord_val:.2f}")
                    else:
                        coord_info.append(f"{dim}={coord_val}")
                else:
                    coord_info.append(f"{dim}[{idx}]")

            self.coord_status.value = f"<b>Current Position:</b> {' | '.join(coord_info)}"
        else:
            self.coord_status.value = "<b>Current Position:</b> --"

    def update_plot(self, change=None):
        """Update the plot"""
        if not self.current_var:
            return

        image_bytes = self.create_plot()
        if image_bytes:
            self.plot_widget.value = image_bytes

    def update_plot_type_options(self, var_name):
        """Update plot type options based on variable capabilities"""
        if not var_name:
            return

        validation = self.validate_plot_types_for_variable(var_name)

        # Create new options with validation info
        new_options = []
        valid_values = []

        plot_type_labels = {
            'map': 'Geographic Map',
            'depth': 'Depth Profile',
            'time': 'Time Series',
            'hovmoller': 'Hovmöller'
        }

        for plot_type, label in plot_type_labels.items():
            if validation[plot_type]:
                new_options.append((label, plot_type))
                valid_values.append(plot_type)
            else:
                new_options.append((f"{label} (not available)", f"{plot_type}_disabled"))

        # Store current value if valid
        current_value = self.plot_type.value
        if current_value not in valid_values:
            # Switch to first available option
            current_value = valid_values[0] if valid_values else 'map'

        # Update the widget
        self.plot_type.options = new_options
        self.plot_type.value = current_value

    def on_var_change(self, change):
        """Handle variable change"""
        self.current_var = change['new']
        self.dimension_names = self.get_dimension_names(self.standard_mapping)

        # Update available plot types for this variable
        self.update_plot_type_options(self.current_var)

        # Create controls with validation
        self.create_dimension_controls(self.current_var)

        # Update range inputs if in manual mode
        if not self.auto_range_checkbox.value:
            self.update_range_inputs_from_data()

        self.update_plot()

    def get_best_initial_plot_type(self, var_name):
        """Determine the best initial plot type for a variable"""
        if not var_name:
            return 'map'

        # Get available plot types
        plot_types = self.validate_plot_types_for_variable(var_name)

        # Prefer map if available, otherwise time series
        if plot_types['map']:
            return 'map'
        elif plot_types['time']:
            return 'time'
        elif plot_types['depth']:
            return 'depth'
        elif plot_types['hovmoller']:
            return 'hovmoller'
        else:
            return 'map'  # Default fallback

    def show(self):
        """Display the viewer"""
        display(self.interface)

        # Initialize with first variable and validate plot types
        if self.vars:
            first_var = self.vars[0]

            self.current_var = first_var
            self.update_plot_type_options(self.current_var)

            # Determine and set the best plot type
            best_plot_type = self.get_best_initial_plot_type(self.current_var)
            self.plot_type.value = best_plot_type

            self.create_dimension_controls(self.current_var)

            # Update range inputs if in manual mode
            if not self.auto_range_checkbox.value:
                self.update_range_inputs_from_data()

            # Generate the initial plot
            self.update_plot()

    def close(self):
        """Close dataset"""
        self.ds.close()


def pyview(filepath, dark_mode=False, dim_mapping=None):
    """
    Create and show ncview-style viewer

    Parameters:
    -----------
    filepath : str
        Path to the NetCDF file
    dark_mode : bool, optional
        Enable dark mode theme (default: False)
    dim_mapping : dict, optional
        Manual dimension mapping, e.g.:
        {'lat': 'yt_ocean', 'lon': 'xt_ocean', 'depth': 'st_ocean', 'time': 'time'}

    Examples:
    ---------
    # Auto-detect dimensions
    viewer = pyview('data.nc')

    # Custom dimension mapping
    viewer = pyview('data.nc', dim_mapping={
        'lat': 'yt_ocean',
        'lon': 'xt_ocean',
        'depth': 'st_ocean'
    })

    # Show detected dimensions
    viewer.show_dimension_mapping()
    """
    viewer = PyView(filepath, dark_mode=dark_mode, dim_mapping=dim_mapping)
    viewer.show()
    return viewer
