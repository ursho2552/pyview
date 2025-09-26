import ipywidgets as widgets
import matplotlib.figure as mfig
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.axes import Axes
from typing import Optional, Any, TYPE_CHECKING

import xarray as xr
import numpy as np
import io

if TYPE_CHECKING:
    from .data_handler import DataHandler

class UIComponents:
    """
    UIComponents class to create and manage the ncview-style interface
    """

    def __init__(self, data_handler: 'DataHandler') -> None:
        """
        Initialize UI components with a DataHandler instance

        Parameters:
        data_handler (DataHandler): Instance of DataHandler for data operations

        Returns:
        None
        """
        self.data_handler: 'DataHandler' = data_handler
        self.current_var: Optional[str] = None
        self.dim_indices: dict[str, int] = {}
        self.coord_displays: dict[str, widgets.Text] = {}
        self.nav_buttons: dict[str, dict[str, widgets.Button]] = {}

        # Storage for widgets
        self.sliders: dict[str, widgets.Widget] = {}

        # Widget attributes (set in setup_interface)
        self.var_selector: widgets.Dropdown
        self.plot_type: widgets.RadioButtons
        self.cmap_selector: widgets.Dropdown
        self.auto_range_checkbox: widgets.Checkbox
        self.vmin_input: widgets.Text
        self.vmax_input: widgets.Text
        self.range_controls: widgets.HBox
        self.dim_controls: widgets.VBox
        self.stats_display: widgets.HTML
        self.plot_widget: widgets.Image
        self.nav_controls: widgets.HBox
        self.coord_status: widgets.HTML
        self.interface: widgets.VBox

        # Create the interface
        self.setup_interface()

    def setup_interface(self):
        """
        Setup the ncview-style interface

        Returns:
        None
        """

        # Variable selector
        self.var_selector = widgets.Dropdown(
            options=[(f"{var}", var) for var in self.data_handler.vars],
            description='Variable:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='290px')
        )
        self.var_selector.observe(self.on_var_change, names='value')


        # Plot type selector (like ncview's view menu)
        self.plot_type = widgets.RadioButtons(
            options=[('Geographic Map', 'map'), ('Depth Profile', 'depth'),
                     ('Time Series', 'time'), ('Hovmöller', 'hovmoller')],
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

        # Navigation controls container - match width of left Variable&View panel
        self.nav_controls = widgets.HBox(
            layout=widgets.Layout(
                justify_content='flex-start',
                width='720px',  # Match width of left Variable&View panel...
                margin='0'
            )
        )

        # Current coordinates display
        self.coord_status = widgets.HTML(
            value="<b>Current Position:</b> Select a variable to begin",
            layout=widgets.Layout(width='800px')
        )

        status_area = widgets.HBox([
            self.coord_status
        ], layout=widgets.Layout(justify_content='flex-start', padding='5px'))

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

    def initialize_first_variable(self) -> None:
        """
        Initialize with first variable

        Returns:
        None
        """
        if self.data_handler.vars:
            first_var = self.data_handler.vars[0]
            self.current_var = first_var
            self.var_selector.value = first_var

            # Call variable change to get a first plot
            self.update_plot_type_options(self.current_var)
            self.create_dimension_controls(self.current_var)

            if not self.auto_range_checkbox.value:
                self.update_range_inputs_from_data()

            self.update_plot()

    def create_dimension_controls(self, var_name: str) -> None:
        """
        Create dimension controls for a variable

        Parameters:
        var_name (str): Variable name for which to create controls

        Returns:
        None
        """
        var = self.data_handler.ds[var_name]
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

    def get_dimension_names(self, dim_mapping: Optional[dict[str, str]] = None) -> dict[str, Optional[str]]:
        """
        Delegate to data_handler

        Parameters:
        dim_mapping (dict, optional): Custom dimension mapping

        Returns:
        dict: Dimension names
        """
        return self.data_handler.get_dimension_names(dim_mapping)

    def get_dimension_name(self, dim_type: str) -> Optional[str]:
        """
        Delegate to data_handler

        Parameters:
        dim_type (str): One of 'lat', 'lon', 'depth', 'time'

        Returns:
        str: Dimension name or None
        """
        return self.data_handler.get_dimension_name(dim_type)

    def validate_plot_types_for_variable(self, var_name: str) -> dict[str, bool]:
        """
        Delegate to data_handler

        Parameters:
        var_name (str): Variable name to validate

        Returns:
        dict: Availability of plot types
        """
        return self.data_handler.validate_plot_types_for_variable(var_name)

    def get_navigable_dimensions_for_plot_type(self, var_name: str, plot_type: str) -> set[str]:
        """
        Delegate to data_handler

        Parameters:
        var_name (str): Variable name
        plot_type (str): One of 'map', 'depth', 'time', 'hovmoller'

        Returns:
        set: Set of dimension names that should be navigable
        """
        return self.data_handler.get_navigable_dimensions_for_plot_type(var_name, plot_type)

    def get_plot_data(self) -> tuple[Optional[xr.DataArray], Optional[list[str]], dict[str, int]]:
        """
        Get plot data from data_handler

        Returns:
        tuple: (data_slice, plot_dims, slice_dict) or None if not available
        """
        # Set current state in data_handler
        self.data_handler.current_var = self.current_var
        self.data_handler.plot_type = self.plot_type.value
        self.data_handler.dim_indices = self.dim_indices

        return self.data_handler.get_plot_data()

    def create_map_plot(self, plot_dims: list[str], ax: Axes, fig: mfig.Figure,
                       values: np.ndarray, vmin: float, vmax: float) -> None:
        """
        Create a geographic map plot

        Parameters:
        plot_dims (list): List of dimension names for the plot
        ax (matplotlib.axes.Axes): Axes to plot on
        fig (matplotlib.figure.Figure): Figure object
        values (np.ndarray): Data values to plot
        vmin (float): Minimum value for color scale
        vmax (float): Maximum value for color scale

        Returns:
        None
        """

        lat_dim, lon_dim = plot_dims
        lats = self.data_handler.ds.coords[lat_dim].values
        lons = self.data_handler.ds.coords[lon_dim].values

        im = ax.pcolormesh(lons, lats, values, cmap=self.cmap_selector.value,
                            shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{self.current_var} - Geographic Map')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(self.current_var)

    def create_depth_profile_plot(self, plot_dims: list[str], ax: Axes,
                                 values: np.ndarray) -> None:
        """
        Create a depth profile plot

        Parameters:
        plot_dims (list): List of dimension names for the plot
        ax (matplotlib.axes.Axes): Axes to plot on
        values (np.ndarray): Data values to plot

        Returns:
        None
        """

        depth_dim = plot_dims[0]
        depths = self.data_handler.ds.coords[depth_dim].values

        ax.plot(values, depths, 'b-', linewidth=2)
        ax.set_xlabel(self.current_var)
        ax.set_ylabel('Depth')
        ax.set_title(f'{self.current_var} - Depth Profile')
        ax.invert_yaxis()  # Depth increases downward
        ax.grid(True, alpha=0.3)

    def create_time_series_plot(self, plot_dims: list[str], ax: Axes,
                               values: np.ndarray) -> None:
        """
        Create a time series plot

        Parameters:
        plot_dims (list): List of dimension names for the plot
        ax (matplotlib.axes.Axes): Axes to plot on
        values (np.ndarray): Data values to plot

        Returns:
        None
        """

        time_dim = plot_dims[0]
        time_coords = self.data_handler.ds.coords[time_dim].values

        ax.plot(time_coords, values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel(self.current_var)
        ax.set_title(f'{self.current_var} - Time Series')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    def create_hovmoller_plot(self, plot_dims: list[str], ax: Axes, fig: mfig.Figure,
                             values: np.ndarray, vmin: float, vmax: float) -> None:
        """
        Create a Hovmöller plot (spatial dimension vs depth)

        Parameters:
        plot_dims (list): List of dimension names for the plot
        ax (matplotlib.axes.Axes): Axes to plot on
        fig (matplotlib.figure.Figure): Figure object
        values (np.ndarray): Data values to plot
        vmin (float): Minimum value for color scale
        vmax (float): Maximum value for color scale

        Returns:
        None
        """

        depth_dim, spatial_dim = plot_dims
        depths = self.data_handler.ds.coords[depth_dim].values
        spatial_coords = self.data_handler.ds.coords[spatial_dim].values

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

    def create_plot(self) -> Optional[bytes]:
        """
        Create plot using matplotlib

        Returns:
        Optional[bytes]: PNG image bytes of the plot or None on error
        """
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
                self.create_map_plot(plot_dims, ax, fig, values, vmin, vmax)

            elif plot_type == 'depth' and len(plot_dims) == 1:
                self.create_depth_profile_plot(plot_dims, ax, values)

            elif plot_type == 'time' and len(plot_dims) == 1:
                self.create_time_series_plot(plot_dims, ax, values)

            elif plot_type == 'hovmoller' and len(plot_dims) == 2:
                self.create_hovmoller_plot(plot_dims, ax, fig, values, vmin, vmax)

            else:
                ax.text(0.5, 0.5, f'View type "{plot_type}" not implemented\nfor this data structure',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)

            # Add slice information to title
            if slice_dict:
                slice_info = []
                for dim, idx in slice_dict.items():
                    if dim in self.data_handler.ds.coords:
                        coord_val = self.data_handler.ds.coords[dim].values[idx]
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

    def update_plot(self, change: Optional[dict[str, Any]] = None) -> None:
        """
        Update the plot

        Parameters:
        change (dict, optional): Change event from widget observers

        Returns:
        None
        """
        if not self.current_var:
            return

        image_bytes = self.create_plot()
        if image_bytes:
            self.plot_widget.value = image_bytes

    def on_range_mode_change(self, change: dict[str, Any]) -> None:
        """
        Handle auto/manual range mode change

        Parameters:
        change (dict): Change event from widget observer

        Returns:
        None
        """

        auto_mode = change['new']

        # Enable/disable manual range inputs
        self.vmin_input.disabled = auto_mode
        self.vmax_input.disabled = auto_mode

        if not auto_mode and self.current_var:
            # Switch to manual mode - set current data range as starting values
            self.update_range_inputs_from_data()

        # Update plot
        self.update_plot()

    def on_range_input_change(self, change: dict[str, Any]) -> None:
        """
        Handle manual range input changes (text to float conversion)

        Parameters:
        change (dict): Change event from widget observer

        Returns:
        None
        """
        # Only update plot if we're in manual mode
        if not self.auto_range_checkbox.value:
            self.update_plot()

    def get_manual_range(self) -> tuple[Optional[float], Optional[float]]:
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

    def update_range_inputs_from_data(self) -> None:
        """
        Update range inputs with current data range

        Returns:
        None
        """
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

    def navigate_dimension(self, dim: str, direction: int) -> None:
        """
        Navigate dimension with wrap-around

        Parameters:
        dim (str): Dimension name
        direction (int): +1 for next, -1 for previous

        Returns:
        None
        """
        if dim not in self.dim_indices:
            return

        current_idx = self.dim_indices[dim]
        max_idx = self.data_handler.ds.sizes[dim] - 1

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

    def update_index_display(self, dim: str) -> None:
        """
        Update the index display (1/N format)

        Parameters:
        dim (str): Dimension name

        Returns:
        None
        """
        current_idx = self.dim_indices[dim]
        total_size = self.data_handler.ds.sizes[dim]

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

    def update_coord_display(self, dim: str, index: int) -> None:
        """
        Update coordinate input field

        Parameters:
        dim (str): Dimension name
        index (int): Current index

        Returns:
        None
        """
        try:
            if dim in self.data_handler.ds.coords:
                coord_val = self.data_handler.ds.coords[dim].values[index]
                if isinstance(coord_val, (int, float, np.number)):
                    self.coord_displays[dim].value = f"{coord_val:.2f}"
                else:
                    self.coord_displays[dim].value = str(coord_val)
            else:
                self.coord_displays[dim].value = str(index)
        except:
            self.coord_displays[dim].value = str(index)

    def on_coord_input_submit(self, widget: widgets.Text, dim: str) -> None:
        """
        Handle coordinate input field submission (Enter key)

        Parameters:
        widget (widgets): The Text widget
        dim (str): Dimension name

        Returns:
        None
        """
        try:
            user_input = widget.value.strip()
            if not user_input:
                return

            if dim in self.data_handler.ds.coords:
                # Try to find the closest coordinate value
                coord_values = self.data_handler.ds.coords[dim].values

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
            dim_size = self.data_handler.ds.dims[dim]
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

    def on_plot_type_change(self, change: dict[str, Any]) -> None:
        """
        Handle plot type change

        Parameters:
        change (dict): Change event from widget observer

        Returns:
        None
        """
        plot_type = change['new']

        # Skip disabled options
        if plot_type.endswith('_disabled'):
            return

        if not self.current_var:
            return

        # Recreate dimension controls with new navigation validation
        self.create_dimension_controls(self.current_var)

        var = self.data_handler.ds[self.current_var]
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
                        mid_idx = len(self.data_handler.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)
                elif dim == lon_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.data_handler.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)

        elif plot_type == 'time':
            # For time series, set to specific location and depth
            for dim in dims:
                if dim == lat_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.data_handler.ds.coords[dim]) // 2
                        self.dim_indices[dim] = mid_idx
                        self.update_index_display(dim)
                        self.update_coord_display(dim, mid_idx)
                elif dim == lon_dim:
                    if dim in self.dim_indices:
                        mid_idx = len(self.data_handler.ds.coords[dim]) // 2
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

    def update_stats(self, values: np.ndarray) -> None:
        """
        Update statistics display

        Parameters:
        values (np.ndarray): Data values to compute statistics on

        Returns:
        None
        """
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

    def update_coord_status(self, slice_dict: Optional[dict[str, int]]) -> None:
        """
        Update current coordinate status

        Parameters:
        slice_dict (dict, optional): Current slice indices for each dimension

        Returns:
        None
        """

        if slice_dict:
            coord_info = []
            for dim, idx in slice_dict.items():
                if dim in self.data_handler.ds.coords:
                    coord_val = self.data_handler.ds.coords[dim].values[idx]
                    if isinstance(coord_val, (int, float, np.number)):
                        coord_info.append(f"{dim}={coord_val:.2f}")
                    else:
                        coord_info.append(f"{dim}={coord_val}")
                else:
                    coord_info.append(f"{dim}[{idx}]")

            self.coord_status.value = f"<b>Current Position:</b> {' | '.join(coord_info)}"
        else:
            self.coord_status.value = "<b>Current Position:</b> --"

    def update_plot_type_options(self, var_name: str) -> None:
        """
        Update plot type options based on variable capabilities

        Parameters:
        var_name (str): Variable name

        Returns:
        None
        """

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

    def on_var_change(self, change: dict[str, Any]) -> None:
        """
        Handle variable change

        Parameters:
        change (dict): Change event from widget observer

        Returns:
        None
        """

        self.current_var = change['new']

        # Update available plot types for this variable
        self.update_plot_type_options(self.current_var)

        # Create controls with validation
        self.create_dimension_controls(self.current_var)

        # Update range inputs if in manual mode
        if not self.auto_range_checkbox.value:
            self.update_range_inputs_from_data()

        self.update_plot()

    def get_best_initial_plot_type(self, var_name: str) -> str:
        """
        Determine the best initial plot type for a variable

        Parameters:
        var_name (str): Variable name

        Returns:
        str: Best plot type ('map', 'depth', 'time', 'hovmoller')
        """

        if not var_name:
            return 'map'

        # Get available plot types
        plot_types = self.validate_plot_types_for_variable(var_name)

        # Priority: map > time > depth > hovmoller
        plot_type = 'map'
        if plot_types['map']:
            plot_type = 'map'
        elif plot_types['time']:
            plot_type = 'time'
        elif plot_types['depth']:
            plot_type = 'depth'
        elif plot_types['hovmoller']:
            plot_type = 'hovmoller'

        return plot_type