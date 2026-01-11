import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvista import themes
from pyvista.core.celltype import CellType
from pyvistaqt import QtInteractor

__all__ = ["plot_mesh", "plot_results"]


ELEMENTS_NODES_TO_VTK = {
    3: CellType.TRIANGLE,
    4: CellType.QUAD,
    6: CellType.QUADRATIC_TRIANGLE,
    8: CellType.QUADRATIC_QUAD,
    9: CellType.LAGRANGE_QUADRILATERAL,
}

import matplotlib.pyplot as plt


class BladeGeometryVisualizer:
    """Class for visualizing wind turbine blade geometric parameters."""

    def __init__(self, blade_definition):
        self.blade_definition = blade_definition

    def plot_chord_distribution(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot chord length distribution along the blade span.

        Returns
        -------
        fig : plt.Figure
            Figure object containing the plot
        ax : plt.Axes
            Axes object containing the plot elements

        Examples
        --------
        >>> visualizer = BladeGeometryVisualizer()
        >>> fig, ax = visualizer.plot_chord_distribution()
        """
        fig, ax = plt.subplots()
        span = self.blade_definition.definition.span
        chord = self.blade_definition.definition.chord

        ax.plot(span, chord, "b-", linewidth=2)
        ax.set_title("Chord Distribution Along Blade Span")
        ax.set_xlabel("Normalized Span Position [-]", fontsize=10)
        ax.set_ylabel("Chord Length [m]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        return fig, ax

    def plot_twist_distribution(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot twist angle distribution along the blade span.

        Returns
        -------
        fig : plt.Figure
            Figure object containing the plot
        ax : plt.Axes
            Axes object containing the plot elements
        """
        fig, ax = plt.subplots()
        span = self.blade_definition.definition.span
        twist = self.blade_definition.definition.degreestwist

        ax.plot(span, twist, "r--", linewidth=2)
        ax.set_title("Twist Distribution Along Blade Span")
        ax.set_xlabel("Normalized Span Position [-]", fontsize=10)
        ax.set_ylabel("Twist Angle [deg]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        return fig, ax

    def plot_thickness_to_chord_ratio(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot thickness-to-chord ratio distribution along the blade span.

        Returns
        -------
        fig : plt.Figure
            Figure object containing the plot
        ax : plt.Axes
            Axes object containing the plot elements
        """
        fig, ax = plt.subplots()
        span = self.blade_definition.definition.span
        thickness = self.blade_definition.definition.percentthick

        ax.plot(span, thickness, "g-.", linewidth=2)
        ax.set_title("Thickness-to-Chord Ratio Distribution")
        ax.set_xlabel("Normalized Span Position [-]", fontsize=10)
        ax.set_ylabel("Thickness/Chord Ratio [-]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        return fig, ax

    def plot_airfoil_type_distribution(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot airfoil type distribution along the blade span using station data.

        Returns
        -------
        fig : plt.Figure
            Figure object containing the plot
        ax : plt.Axes
            Axes object containing the plot elements

        Notes
        -----
        - Uses actual station data from blade definition
        - Creates a categorical plot with airfoil reference labels
        - Maintains original span positions from blade stations
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data from blade stations
        stations = self.blade_definition.definition.stations
        span_positions = [s.spanlocation for s in stations]
        airfoil_refs = [s.airfoil.reference for s in stations]

        # Create numerical mapping for airfoil types
        unique_airfoils = list(dict.fromkeys(airfoil_refs))  # Preserve order
        airfoil_id_map = {ref: idx for idx, ref in enumerate(unique_airfoils)}
        airfoil_ids = [airfoil_id_map[ref] for ref in airfoil_refs]

        # Create scatter plot with text annotations
        scatter = ax.scatter(
            span_positions, airfoil_ids, c=airfoil_ids, cmap="tab20", s=100, edgecolor="k", zorder=3
        )

        # Configure axes and labels
        ax.set_title("Airfoil Type Distribution Along Blade Span", fontsize=14)
        ax.set_xlabel("Normalized Span Position [-]", fontsize=12)
        ax.set_ylabel("Airfoil Type", fontsize=12)
        ax.set_yticks(list(airfoil_id_map.values()))
        ax.set_yticklabels(list(airfoil_id_map.keys()))
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add span position markers
        ax.vlines(
            span_positions,
            ymin=min(airfoil_ids) - 0.5,
            ymax=max(airfoil_ids) + 0.5,
            colors="lightgray",
            linestyles="dotted",
            zorder=1,
        )

        # Add legend with color mapping
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab20(i / len(unique_airfoils)),
                markersize=10,
                label=f"{ref} (ID: {i})",
            )
            for i, ref in enumerate(unique_airfoils)
        ]

        ax.legend(
            handles=legend_elements,
            title="Airfoil Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
        )
        plt.tight_layout()
        return fig, ax

    def plot_pitch_axis_position(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot pitch axis position distribution along the blade span.

        Returns
        -------
        fig : plt.Figure
            Figure object containing the plot
        ax : plt.Axes
            Axes object containing the plot elements
        """
        fig, ax = plt.subplots()
        span = self.blade_definition.definition.span
        pitch_axis = self.blade_definition.definition.aerocenter

        ax.plot(span, pitch_axis, "c^", markersize=6, markeredgecolor="k")
        ax.set_title("Aerodynamic Center Distribution")
        ax.set_xlabel("Normalized Span Position [-]", fontsize=10)
        ax.set_ylabel("Aerodynamic Center [% chord]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, ax


class MeshViewer(QWidget):
    """A widget for visualizing finite element meshes with node and element sets.

    Attributes
    ----------
    mesh : Any
        The input mesh data containing nodes, elements, and sets.
    current_grid : pyvista.UnstructuredGrid or None
        The currently displayed grid in the plotter.
    is_2D_mesh : bool
        Flag indicating if the mesh is 2D (all Z coordinates are zero).
    plotter : pyvistaqt.QtInteractor
        The PyVista Qt interactor for 3D visualization.
    """

    def __init__(self, mesh) -> None:
        """Initialize the MeshViewer with given mesh data.

        Parameters
        ----------
        mesh : Any
            The mesh object containing nodes, elements, and sets information.
        """
        super().__init__()
        self.mesh = mesh
        self.current_grid = None
        self.is_2D_mesh = not bool(self.mesh.coords_array[:, 2].sum())
        # Visualization colors (start from plotter defaults later)
        self.background_color = None
        self.mesh_color = (0.83, 0.83, 0.85)
        self.edge_color = (0.2, 0.2, 0.24)
        self.point_color = (1.0, 0.4, 0.0)
        self.init_ui()
        self.create_set_table()

    def init_ui(self):
        """Initialize the user interface components and layout."""
        self.setWindowTitle("Mesh Viewer")
        self.setMinimumSize(800, 600)

        # Main vertical layout with toolbar at top
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create and add toolbar (kept compact)
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # Content layout using QSplitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Control panel (left)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Visualization mode selector
        control_layout.addWidget(QLabel("Visualization Mode:"))
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["Surface with edges", "Wireframe", "Surface", "Points"])
        self.vis_mode_combo.setToolTip("Select mesh visualization style")
        self.vis_mode_combo.currentIndexChanged.connect(self.update_plot)
        control_layout.addWidget(self.vis_mode_combo)

        # Sets group box
        self.sets_group = QGroupBox("Sets (Nodes and Elements)")
        sets_layout = QVBoxLayout()

        # Set filter
        self.filter_le = QLineEdit()
        self.filter_le.setPlaceholderText("Filter sets by name or type")
        self.filter_le.textChanged.connect(self.filter_table)
        sets_layout.addWidget(self.filter_le)

        # Sets table
        self.sets_table = QTableWidget()
        self.sets_table.setColumnCount(2)
        self.sets_table.setHorizontalHeaderLabels(["Set", "Type"])
        self.sets_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.sets_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sets_table.itemChanged.connect(self.update_plot)
        # Hide row numbers
        self.sets_table.verticalHeader().setVisible(False)
        sets_layout.addWidget(self.sets_table)

        self.sets_group.setLayout(sets_layout)
        control_layout.addWidget(self.sets_group)
        # Make sets group expand to fill space
        control_layout.setStretchFactor(self.sets_group, 1)

        splitter.addWidget(control_widget)

        # PyVista visualization panel (right)
        pv.set_plot_theme(themes.ParaViewTheme())
        self.plotter = QtInteractor(parent=self)
        # Keep default theme background; store it for color dialog
        self.background_color = tuple(self.plotter.background_color)
        self.default_background_color = self.background_color
        splitter.addWidget(self.plotter)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_toolbar(self) -> QWidget:
        """Create the visualization toolbar with view and color controls."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 6, 10, 6)
        toolbar_layout.setSpacing(6)
        toolbar_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar_widget.setMaximumHeight(52)

        # View buttons group
        view_label = QLabel("View:")
        toolbar_layout.addWidget(view_label)

        xy_btn = QPushButton("XY (Top)")
        xy_btn.setToolTip("View from top (XY plane)")
        xy_btn.clicked.connect(self._view_xy_plane)
        toolbar_layout.addWidget(xy_btn)

        xz_btn = QPushButton("XZ (Front)")
        xz_btn.setToolTip("View from front (XZ plane)")
        xz_btn.clicked.connect(self._view_xz_plane)
        toolbar_layout.addWidget(xz_btn)

        yz_btn = QPushButton("YZ (Side)")
        yz_btn.setToolTip("View from side (YZ plane)")
        yz_btn.clicked.connect(self._view_yz_plane)
        toolbar_layout.addWidget(yz_btn)

        iso_btn = QPushButton("Isometric")
        iso_btn.setToolTip("Isometric view")
        iso_btn.clicked.connect(self._view_isometric)
        toolbar_layout.addWidget(iso_btn)

        # Separator
        toolbar_layout.addSpacing(15)

        # Reset and info buttons
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setToolTip("Reset camera to default view")
        reset_btn.clicked.connect(self._reset_view)
        toolbar_layout.addWidget(reset_btn)

        info_btn = QPushButton("ℹ️ Info")
        info_btn.setToolTip("Show mesh information")
        info_btn.clicked.connect(self._show_mesh_info_dialog)
        toolbar_layout.addWidget(info_btn)

        color_btn = QPushButton("🎨 Colors")
        color_btn.setToolTip("Configure visualization colors")
        color_btn.clicked.connect(self._show_colors_dialog)
        toolbar_layout.addWidget(color_btn)

        preset_label = QLabel("Preset:")
        toolbar_layout.addWidget(preset_label)

        preset_combo = QComboBox()
        preset_combo.addItems(["Default", "Dark", "Blueprint"])
        preset_combo.currentTextChanged.connect(self._apply_color_preset)
        preset_combo.setFixedWidth(120)
        toolbar_layout.addWidget(preset_combo)

        # Add stretch to push buttons to the left
        toolbar_layout.addStretch()

        return toolbar_widget

    def _show_mesh_info_dialog(self):
        """Show mesh information in a compact, polished popup dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Mesh Information")
        dialog.setMinimumWidth(380)
        dialog.setMaximumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Mesh Statistics")
        font = title.font()
        font.setPointSize(11)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Count unique nodes by ID to avoid duplicates
        unique_node_ids = {node.id for node in self.mesh.nodes}
        num_nodes = len(unique_node_ids)
        num_elements = len(self.mesh.elements)

        stats = [
            ("Total Nodes", num_nodes),
            ("Total Elements", num_elements),
            ("Node Sets", len(self.mesh.node_sets_names)),
            ("Element Sets", len(self.mesh.element_sets_names)),
        ]

        for row, (label, value) in enumerate(stats):
            key_lbl = QLabel(label)
            val_lbl = QLabel(f"{value:,}")
            key_font = key_lbl.font()
            key_font.setPointSize(10)
            key_lbl.setFont(key_font)
            val_font = val_lbl.font()
            val_font.setPointSize(10)
            val_font.setBold(True)
            val_lbl.setFont(val_font)
            grid.addWidget(key_lbl, row, 0)
            grid.addWidget(val_lbl, row, 1)

        layout.addLayout(grid)

        # Bounding Box Section
        coords = self.mesh.coords_array
        if coords.size > 0:
            layout.addSpacing(10)
            bbox_title = QLabel("Bounding Box")
            bbox_title.setFont(font)
            layout.addWidget(bbox_title)

            bbox_grid = QGridLayout()
            bbox_grid.setHorizontalSpacing(12)
            bbox_grid.setVerticalSpacing(4)

            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            dims = max_c - min_c

            bbox_data = [
                ("X Range", f"[{min_c[0]:.4f}, {max_c[0]:.4f}]", f"Δ={dims[0]:.4f}"),
                ("Y Range", f"[{min_c[1]:.4f}, {max_c[1]:.4f}]", f"Δ={dims[1]:.4f}"),
                ("Z Range", f"[{min_c[2]:.4f}, {max_c[2]:.4f}]", f"Δ={dims[2]:.4f}"),
            ]

            for row, (label, range_str, dim_str) in enumerate(bbox_data):
                l_lbl = QLabel(label)
                r_lbl = QLabel(range_str)
                d_lbl = QLabel(dim_str)
                
                for lbl in [l_lbl, r_lbl, d_lbl]:
                    f = lbl.font()
                    f.setPointSize(9)
                    if lbl is d_lbl:
                        f.setBold(True)
                    lbl.setFont(f)
                
                r_lbl.setStyleSheet("color: #555;")
                
                bbox_grid.addWidget(l_lbl, row, 0)
                bbox_grid.addWidget(r_lbl, row, 1)
                bbox_grid.addWidget(d_lbl, row, 2)

            layout.addLayout(bbox_grid)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dialog.setLayout(layout)
        dialog.exec()

    def _show_colors_dialog(self):
        """Show color configuration dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Color Palette")
        dialog.setMinimumWidth(360)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Customize Viewer Colors")
        font = title.font()
        font.setPointSize(11)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        color_buttons: dict[str, QPushButton] = {}

        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_combo = QComboBox()
        preset_combo.addItems(["Default", "Dark", "Blueprint"])
        preset_combo.setFixedWidth(150)
        preset_combo.currentTextChanged.connect(
            lambda name: self._apply_color_preset(name, color_buttons)
        )
        preset_row.addWidget(preset_label)
        preset_row.addWidget(preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def add_color_row(row: int, label: str, attr: str):
            lbl = QLabel(label)
            btn = QPushButton()
            btn.setFixedSize(44, 28)
            btn.clicked.connect(lambda _, a=attr, b=btn: self._pick_color(a, b))
            btn.setStyleSheet(self._color_button_style(getattr(self, f"{attr}_color", (1, 1, 1))))
            color_buttons[attr] = btn
            grid.addWidget(lbl, row, 0)
            grid.addWidget(btn, row, 1)

        add_color_row(0, "Background", "background")
        add_color_row(1, "Mesh", "mesh")
        add_color_row(2, "Edges", "edge")
        add_color_row(3, "Points", "point")

        layout.addLayout(grid)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dialog.setLayout(layout)
        dialog.exec()

    def _color_button_style(self, rgb_tuple: tuple[float, float, float]) -> str:
        r, g, b = (int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))
        return f"background-color: rgb({r}, {g}, {b}); border: 1px solid #777; border-radius: 4px;"

    def _pick_color(self, color_type: str, button: QPushButton):
        """Open color picker for the specified color type and apply instantly."""
        color = QColorDialog.getColor()
        if not color.isValid():
            return

        rgb = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)

        if color_type == "background":
            self.background_color = rgb
            self.plotter.set_background(rgb)
        elif color_type == "mesh":
            self.mesh_color = rgb
        elif color_type == "edge":
            self.edge_color = rgb
        elif color_type == "point":
            self.point_color = rgb

        button.setStyleSheet(self._color_button_style(rgb))
        self.update_plot()

    def _apply_color_preset(self, preset_name: str, buttons: dict[str, QPushButton] | None = None):
        """Apply a named color preset to mesh, edges, points, and background."""

        presets = {
            "Default": {
                "background": getattr(self, "default_background_color", self.background_color),
                "mesh": (0.83, 0.85, 0.90),
                "edge": (0.2, 0.2, 0.2),
                "point": (1.0, 0.4, 0.0),
            },
            "Dark": {
                "background": (0.08, 0.09, 0.11),
                "mesh": (0.76, 0.78, 0.84),
                "edge": (0.32, 0.34, 0.40),
                "point": (0.98, 0.54, 0.20),
            },
            "Blueprint": {
                "background": (0.03, 0.08, 0.20),
                "mesh": (0.72, 0.78, 0.90),
                "edge": (0.36, 0.50, 0.80),
                "point": (1.0, 0.75, 0.35),
            },
        }

        palette = presets.get(preset_name)
        if not palette:
            return

        self.background_color = palette["background"]
        self.mesh_color = palette["mesh"]
        self.edge_color = palette["edge"]
        self.point_color = palette["point"]

        self.plotter.set_background(self.background_color)

        if buttons:
            for key, btn in buttons.items():
                btn.setStyleSheet(self._color_button_style(getattr(self, f"{key}_color")))

        self.update_plot()

    def _view_xy_plane(self):
        """Set camera view to XY plane (top)."""
        self.plotter.view_xy()
        self.plotter.render()

    def _view_xz_plane(self):
        """Set camera view to XZ plane (front)."""
        self.plotter.view_xz()
        self.plotter.render()

    def _view_yz_plane(self):
        """Set camera view to YZ plane (side)."""
        self.plotter.view_yz()
        self.plotter.render()

    def _view_isometric(self):
        """Set camera to isometric view."""
        self.plotter.view_isometric()
        self.plotter.render()

    def _reset_view(self):
        """Reset the camera to default view."""
        self.plotter.reset_camera()
        self.plotter.render()

    def create_set_table(self):
        """Populate the table with node and element sets from the mesh."""
        node_sets = self.mesh.node_sets_names
        element_sets = self.mesh.element_sets_names
        total_rows = len(node_sets) + len(element_sets)

        self.sets_table.blockSignals(True)
        self.sets_table.setRowCount(total_rows)
        row = 0
        # Add node sets
        for set_name in node_sets:
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setText(set_name)
            type_item = QTableWidgetItem("Node")
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.sets_table.setItem(row, 0, chk_item)
            self.sets_table.setItem(row, 1, type_item)
            row += 1
        # Add element sets
        for set_name in element_sets:
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setText(set_name)
            type_item = QTableWidgetItem("Element")
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.sets_table.setItem(row, 0, chk_item)
            self.sets_table.setItem(row, 1, type_item)
            row += 1

        self.sets_table.blockSignals(False)

    def filter_table(self, text: str):
        """Filter the sets table based on input text.

        Parameters
        ----------
        text : str
            The filter text to match against set names and types.
        """
        text = text.lower()
        for row in range(self.sets_table.rowCount()):
            name = self.sets_table.item(row, 0).text().lower()
            set_type = self.sets_table.item(row, 1).text().lower()
            self.sets_table.setRowHidden(row, text not in name and text not in set_type)

    def get_selected_sets(self) -> tuple[list[str], list[str]]:
        """Get currently selected sets from the table.

        Returns
        -------
        tuple[list[str], list[str]]
            Two lists containing selected node set names and element set names.
        """
        node_sets, element_sets = [], []
        for row in range(self.sets_table.rowCount()):
            if (
                not self.sets_table.isRowHidden(row)
                and self.sets_table.item(row, 0).checkState() == Qt.Checked
            ):
                set_name = self.sets_table.item(row, 0).text()
                if self.sets_table.item(row, 1).text() == "Node":
                    node_sets.append(set_name)
                else:
                    element_sets.append(set_name)
        return node_sets, element_sets

    def _create_element_grid(
        self, selected_elements: set[int] = None
    ) -> pv.UnstructuredGrid | None:
        """Create PyVista grid from specified element IDs.

        Parameters
        ----------
        element_ids : set of int, optional
            IDs of elements to include in the grid. If None, includes all elements.

        Returns
        -------
        pyvista.UnstructuredGrid or None
            Constructed unstructured grid, or None if no valid elements found.
        """
        cells, cell_types = [], []
        visible_elements = selected_elements or self.mesh.elements
        for element in visible_elements:
            if element:
                cells.append([len(element.node_ids)] + list(element.node_ids))
                cell_types.append(ELEMENTS_NODES_TO_VTK[len(element.node_ids)])
        return (
            pv.UnstructuredGrid(
                np.hstack(cells).astype(np.int32),
                np.array(cell_types),
                np.array(self.mesh.coords_array),
            )
            if cells
            else None
        )

    def _create_node_points(self, node_ids: set[int]) -> pv.PolyData | None:
        """Create point cloud from specified node IDs.

        Parameters
        ----------
        node_ids : set of int
            IDs of nodes to include in the point cloud.

        Returns
        -------
        pyvista.PolyData or None
            Points data or None if no valid nodes found.
        """
        if not node_ids:
            return None

        nodes = np.array(self.mesh.coords_array)
        valid_ids = [i for i in node_ids if i < len(nodes)]
        return pv.PolyData(nodes[valid_ids]) if valid_ids else None

    def _add_custom_axes(self, label_size=20):
        """Add customized axes to the plotter.

        Parameters
        ----------
        label_size : int, optional
            Font size for axis labels, by default 20
        """
        self.plotter.show_axes()

    @Slot()
    def update_plot(self):
        """Update the 3D visualization based on current selections."""
        self.plotter.clear()

        # Set visualization style
        vis_mode = self.vis_mode_combo.currentText()
        style = {}
        if vis_mode == "Wireframe":
            style = {"style": "wireframe", "color": self.mesh_color}
        elif vis_mode == "Points":
            style = {
                "style": "points",
                "render_points_as_spheres": True,
                "point_size": 10,
                "color": self.mesh_color,
            }
        elif "Surface" in vis_mode:
            style = {"style": "surface", "color": self.mesh_color}
            if "edges" in vis_mode:
                style.update({"show_edges": True, "edge_color": self.edge_color})

        # Get selected sets
        node_sets, element_sets = self.get_selected_sets()

        # Process elements
        if element_sets:
            for s in element_sets:
                element_set = self.mesh.get_element_set(s)

            if grid := self._create_element_grid(selected_elements=element_set.elements):
                self.plotter.add_mesh(grid, name="elements", **style)

        # Process nodes
        if node_sets:
            node_ids = set()
            for s in node_sets:
                node_set = self.mesh.get_node_set(s)
                node_ids.update(node_set.node_ids)

            if points := self._create_node_points(node_ids):
                self.plotter.add_points(
                    points,
                    name="nodes",
                    **{
                        "style": "points",
                        "point_size": 10,
                        "render_points_as_spheres": True,
                        "color": self.point_color,
                    },
                )

        # Show full mesh if no selections
        if not element_sets and not node_sets:
            if grid := self._create_element_grid():
                self.plotter.add_mesh(grid, name="surface", **style)

        self._configure_view()
        self.plotter.render()

    def _configure_view(self):
        if self.is_2D_mesh:
            self.plotter.view_xy()
            self.plotter.enable_image_style()
        else:
            self.plotter.view_xz()

        light = pv.Light(position=(0, 0, 1), light_type="camera light")
        self._add_custom_axes()
        self.plotter.add_light(light)
        self.plotter.reset_camera()


def plot_mesh(mesh):
    """Launch the mesh visualization application.

    Parameters
    ----------
    mesh : Any
        The mesh object to visualize.

    Returns
    -------
    None
    """
    app = QApplication.instance() or QApplication()
    viewer = MeshViewer(mesh)
    viewer.update_plot()
    viewer.show()
    app.exec()
    return None


class SimulationViewer(MeshViewer):
    """Widget for visualizing FEM meshes and simulation results.

    Inherits from MeshViewer and adds result field visualization capabilities.

    Attributes
    ----------
    point_data : dict
        Dictionary containing nodal data {name: array}
    warp_vector : str or None
        Currently selected vector field for warping
    warp_scale : float
        Current scaling factor for vector warping
    """

    def __init__(self, mesh, point_data) -> None:
        """Initialize the viewer with mesh and result data.

        Parameters
        ----------
        mesh : object
            Mesh object containing nodes/elements information
        point_data : dict
            Dictionary of nodal data {name: numpy array}
        """
        super().__init__(mesh)
        self.point_data = point_data
        self.warp_vector = None
        self.warp_scale = 0.0
        self._add_result_controls()

    def _add_result_controls(self):
        """Add result visualization controls to the UI."""
        control_layout = self.findChild(QVBoxLayout)

        # Scalar field selector
        control_layout.insertWidget(2, QLabel("Scalar Field:"))
        self.field_combo = QComboBox()
        self.field_combo.addItem("None")
        self.field_combo.addItems(sorted(self.point_data.keys()))
        control_layout.insertWidget(3, self.field_combo)

        # Vector warping controls
        control_layout.insertWidget(4, QLabel("Vector Warping:"))
        self.warp_combo = QComboBox()
        self.warp_combo.addItem("None")
        vector_fields = [
            name
            for name, data in self.point_data.items()
            if data.ndim == 2 and data.shape[1] in (2, 3)
        ]
        self.warp_combo.addItems(vector_fields)
        control_layout.insertWidget(5, self.warp_combo)

        self.warp_slider = QSlider(Qt.Horizontal)
        self.warp_slider.setRange(0, 200)
        self.warp_slider.setValue(0)
        self.warp_scale_label = QLabel("Warp Scale: 0%")
        control_layout.insertWidget(6, self.warp_scale_label)
        control_layout.insertWidget(7, self.warp_slider)

        # Connect signals
        self.field_combo.currentIndexChanged.connect(self.update_plot)
        self.warp_combo.currentTextChanged.connect(self._update_warp_vector)
        self.warp_slider.valueChanged.connect(self._update_warp_scale)

    def _create_element_grid(self, selected_elements=None) -> pv.UnstructuredGrid | None:
        """Create VTK grid with result data and optional warping."""
        grid = super()._create_element_grid(selected_elements)
        if not grid:
            return None

        # Add point data
        for name, data in self.point_data.items():
            grid.point_data[name] = data

        # Apply vector warping
        if self.warp_vector and self.warp_vector in grid.point_data and self.warp_scale != 0:
            factor = self.warp_scale * self._get_auto_scale_factor(grid)
            warped_grid = grid.warp_by_vector(self.warp_vector, factor=factor)

            # Preserve original data and update coordinates
            warped_grid.point_data.update(grid.point_data)
            return warped_grid
        return grid

    def _add_mesh_with_results(self, grid, style, field, name):
        """Add mesh to plotter with appropriate result visualization."""
        if field != "None" and field in grid.point_data:
            self.plotter.add_mesh(
                grid, name=name, scalars=field, scalar_bar_args={"title": field}, **style
            )
        else:
            # Clear any active scalars explicitly
            grid.active_scalars_name = None
            self.plotter.add_mesh(grid, name=name, scalars=None, **style)

    def _get_auto_scale_factor(self, grid) -> float:
        """Calculate automatic scaling factor for visualization."""
        bounds = grid.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )
        return diagonal * 0.05  # 5% of model size

    @Slot()
    def update_plot(self):
        """Update the visualization based on current settings."""
        try:
            self.plotter.clear()
            selected_field = self.field_combo.currentText()

            # Visualization style configuration
            vis_mode = self.vis_mode_combo.currentText()
            style = self._get_visualization_style(vis_mode)

            # Get selected sets
            node_sets, element_sets = self.get_selected_sets()

            # Process element sets
            if element_sets:
                for s in element_sets:
                    element_set = self.mesh.get_element_set(s)

                if grid := self._create_element_grid(element_set.elements):
                    self._add_mesh_with_results(grid, style, selected_field, "elements")

            # Process node sets
            if node_sets:
                node_ids = set()
                for s in node_sets:
                    node_set = self.mesh.get_node_set(s)
                    node_ids.update(node_set.node_ids)

                if points := self._create_node_points(node_ids):
                    self.plotter.add_points(
                        points,
                        name="nodes",
                        **{"style": "points", "point_size": 10, "render_points_as_spheres": True},
                    )

            # Show full mesh if no selections
            if not element_sets and not node_sets:
                if grid := self._create_element_grid():
                    self._add_mesh_with_results(grid, style, selected_field, "surface")

            # Configure view
            self._configure_view()
            self.plotter.render()

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to update plot:\n{str(e)}")

    def _update_warp_vector(self, vector_name):
        """Handle vector field selection changes."""
        self.warp_vector = vector_name if vector_name != "None" else None
        self.update_plot()

    def _update_warp_scale(self, value):
        """Handle warp scale slider changes."""
        self.warp_scale = value / 100.0  # Convert percentage to factor
        self.warp_scale_label.setText(f"Warp Scale: {value}%")
        self.update_plot()
        self.plotter.reset_camera()

    def _get_visualization_style(self, vis_mode: str) -> dict:
        """Determina el estilo de visualización."""
        style = {}
        if vis_mode == "Wireframe":
            style = {"style": "wireframe"}
        elif vis_mode == "Points":
            style = {
                "style": "points",
                "render_points_as_spheres": True,
                "point_size": 10,
            }
        else:  # Surface modes
            style = {"style": "surface"}
            if "edges" in vis_mode:
                style.update({"show_edges": True, "edge_color": "black"})
        return style


def plot_results(mesh, point_data):
    """Launch the results visualization application.

    Parameters
    ----------
    mesh : object
        Mesh object containing nodes/elements information
    point_data : dict
        Dictionary of nodal data {name: numpy array}
    """
    app = QApplication.instance() or QApplication()
    viewer = SimulationViewer(mesh, point_data)
    viewer.update_plot()
    viewer.show()
    app.exec()
