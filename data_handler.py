from typing import Optional
import xarray as xr

# Common dimension keywords for dimension matching
DIMENSION_KEYWORDS = {
    'lat': ['lat', 'yt', 'yu', 'y', 'latitude'],
    'lon': ['lon', 'xt', 'xu', 'x', 'longitude'],
    'depth': ['dep', 'lev', 'z', 'st', 'sw', 'depth', 'level', 'sigma', 's_'],
    'time': ['tim', 'time', 'ocean_time', 'time_counter', 'year', 'month', 'day']
}

class DataHandler:
    """
    DataHandler class to manage NetCDF data and dimension mappings
    """

    def __init__(self, filepath: str, dim_mapping: Optional[dict[str, str]] = None) -> None:
        """
        Initialize DataHandler with dataset and dimension mapping

        Parameters:
        filepath (str): Path to the NetCDF file or files (wildcards allowed)
        dim_mapping (dict, optional): Manual dimension mapping

        Returns:
        None
        """

        self.filepath: str = filepath
        self.ds: xr.Dataset = xr.open_mfdataset(filepath, decode_times=False, data_vars='all')
        self.vars: list[str] = list(self.ds.data_vars)
        self.current_var: Optional[str] = None
        self.plot_type: Optional[str] = None
        self.dim_indices: dict[str, int] = {}
        self.standard_mapping: Optional[dict[str, str]] = dim_mapping
        self.dimension_names: dict[str, Optional[str]] = self.get_dimension_names(dim_mapping=self.standard_mapping)

    def get_dimension_names(self, dim_mapping: Optional[dict[str, str]] = None) -> dict[str, Optional[str]]:
        """
        Get standardized dimension names based on provided mapping or common keywords

        Parameters:
        dim_mapping (dict, optional): User-provided dimension mapping

        Returns:
        dict: Standardized dimension names for 'lat', 'lon', 'depth', 'time
        """

        if dim_mapping is None:
            dim_mapping = {}

        # Default dimension names
        dim_names: dict[str, Optional[str]] = {
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

    def has_dimension_type(self, var_name: str, dim_type: str) -> bool:
        """Check if a variable has a specific dimension type"""
        if not var_name or dim_type not in self.dimension_names:
            return False

        dim_name = self.dimension_names[dim_type]
        if not dim_name:
            return False

        var = self.ds[var_name]
        return dim_name in var.dims

    def get_dimension_name(self, dim_type: str) -> Optional[str]:
        """
        Get the actual dimension name for a dimension type

        Parameters:
        dim_type (str): One of 'lat', 'lon', 'depth', 'time'

        Returns:
        Optional[str]: The dimension name or None if not found

        """
        return self.dimension_names.get(dim_type)

    def validate_plot_types_for_variable(self, var_name: str) -> dict[str, bool]:
        """
        Validate which plot types are available for a given variable

        Parameters:
        var_name (str): Variable name to validate

        Returns:
        dict: Availability of plot types {'map': bool, 'depth': bool, 'time': bool, 'hovmoller': bool}

        """
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

    def get_navigable_dimensions_for_plot_type(self, var_name: str, plot_type: str) -> set[str]:
        """
        Get which dimensions should be navigable for a given plot type

        Parameters:
        var_name (str): Variable name
        plot_type (str): One of 'map', 'depth', 'time', 'hovmoller'

        Returns:
        set: Set of dimension names that should be navigable
        """
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

    def get_plot_data(self) -> tuple[Optional[xr.DataArray], Optional[list[str]], dict[str, int]]:
        """
        Get data for plotting based on current view type

        Returns:
        Union[None, xr.DataArray, tuple[None, None, dict[str, int]]]:
        - For 'map', 'depth', 'time', returns (data_slice, plot_dims, slice_dict)
        - For unsupported or disabled plot types, returns None
        """
        if not self.current_var:
            return None, None, {}

        var = self.ds[self.current_var]

        plot_type = self.plot_type

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

    def close(self) -> None:
        """
        Close dataset

        Returns:
        None
        """

        self.ds.close()
