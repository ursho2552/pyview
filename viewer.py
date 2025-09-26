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

from IPython.display import display
from typing import Optional
from .data_handler import DataHandler
from .ui_components import UIComponents

class PyView:
    """
    PyView class to coordinate between data handling and UI
    """

    def __init__(self, filepath: str, dim_mapping: Optional[dict[str, str]] = None) -> None:
        """
        Initialize PyView with filepath and optional dimension mapping

        Parameters:
        filepath (str): Path to the NetCDF file or files (wildcards allowed)
        dim_mapping (dict, optional): Manual dimension mapping

        Returns:
        None
        """
        self.data_handler: DataHandler = DataHandler(filepath, dim_mapping)
        self.ui: UIComponents = UIComponents(self.data_handler)

    def show(self) -> None:
        """
        Display the viewer

        Returns:
        None
        """
        display(self.ui.interface)
        self.ui.initialize_first_variable()

    def close(self) -> None:
        """
        Close dataset

        Returns:
        None
        """
        self.data_handler.close()

def pyview(filepath: str, dim_mapping: Optional[dict[str, str]] = None) -> PyView:
    """
    Create and show ncview-style viewer

    Parameters:
    filepath (str): Path to the NetCDF file or files (wildcards allowed)
    dim_mapping (dict, optional): Manual dimension mapping

    Returns:
    PyView: Instance of the PyView class

    Examples:
    # Auto-detect dimensions
    viewer = pyview('data.nc')

    # Custom dimension mapping
    viewer = pyview('data.nc', dim_mapping={
        'lat': 'yt_ocean',
        'lon': 'xt_ocean',
        'depth': 'st_ocean'
    })

    """
    viewer = PyView(filepath, dim_mapping=dim_mapping)
    viewer.show()
    return viewer
