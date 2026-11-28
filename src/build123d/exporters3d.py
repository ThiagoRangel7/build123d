"""
build123d 3D Exporters

name: exporters3d.py
by:   Gumyr
date: March 6th, 2024

desc:
    Exporters for 3D models such as STEP.

license:

    Copyright 2024 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

# pylint has trouble with the OCP imports
# pylint: disable=no-name-in-module, import-error

import ctypes
import ctypes.util
import struct
import sys
import tempfile
import threading
import warnings
import webbrowser
import zlib
from datetime import datetime
from functools import cache
from math import ceil
from io import BytesIO
from os import PathLike, fsdecode
from pathlib import Path
from typing import BinaryIO, cast

import OCP.TopAbs as ta
import requests
from anytree import PreOrderIter
from OCP.AIS import AIS_DisplayMode, AIS_InteractiveContext, AIS_Shape
from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
from OCP.Aspect import (
    Aspect_DisplayConnection,
    Aspect_DisplayConnectionDefinitionError,
    Aspect_TypeOfLine,
    Aspect_Window,
)
from OCP.BinTools import BinTools, BinTools_FormatVersion
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Graphic3d import (
    Graphic3d_Camera,
    Graphic3d_MaterialAspect,
    Graphic3d_NameOfMaterial,
)
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.IGESControl import IGESControl_Controller
from OCP.Image import Image_AlienPixMap
from OCP.gp import gp_Ax2, gp_Ax3, gp_Trsf
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity, Message_ProgressRange
from OCP.OpenGl import OpenGl_GraphicDriver
from OCP.Prs3d import Prs3d_LineAspect
from OCP.RWGltf import RWGltf_CafWriter
from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_Controller, STEPControl_StepModelType
from OCP.StlAPI import StlAPI_Writer
from OCP.TCollection import (
    TCollection_AsciiString,
    TCollection_ExtendedString,
    TCollection_HAsciiString,
)
from OCP.collections import (
    IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString,
)
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_FormatVersion
from OCP.V3d import V3d_Viewer
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.XSControl import XSControl_WorkSession

from threejs_materials import PbrProperties, inject_materials

from build123d.build_constants import UNITS_PER_METER
from build123d.build_enums import PrecisionMode, Unit
from build123d.exporters import Export2D, _page, unit_conversion_scale
from build123d.geometry import (
    TOLERANCE,
    BoundBox,
    Color,
    ColorLike,
    Location,
    Vector,
    VectorLike,
)
from build123d.topology import Compound, Curve, Part, Shape, Sketch
from build123d.topology.one_d import _drawing_frame, _drawing_look_at


def _create_xde(
    to_export: Shape, unit: Unit = Unit.MM, auto_naming: bool = False
) -> TDocStd_Document:
    """create_xde

    An OpenCASCADE Technology (OCCT) XDE (eXtended Data Exchange) document is a
    sophisticated framework designed for comprehensive CAD data management,
    facilitating not just the geometric modeling of shapes but also the handling
    of associated metadata, such as material properties, color, layering, and
    annotations. This integrated environment supports the organization and
    manipulation of complex assemblies and parts, enabling users to navigate,
    edit, and query both the structure and attributes of CAD models. XDE's
    architecture is particularly beneficial for applications requiring detailed
    interoperability between different CAD systems, offering robust tools for
    data exchange, visualization, and modification while preserving the integrity
    and attributes of the CAD data across various stages of product development.

    Args:
        to_export (Shape): object or assembly
        unit (Unit, optional): shape units. Defaults to Unit.MM.
        auto_naming (bool, optional): whether to use ShapeTool AutoNaming.
            Defaults to False.

    Returns:
        TDocStd_Document: XDE document
    """
    # Create and initialize the XCAF document. It is built directly, as the
    # importers build theirs, and never opened in the XCAF application:
    # InitDocument is all the writers need, and a document that was never
    # opened leaves nothing behind to close (#1522). NewDocument would open a
    # document in the application and hand it back, which leaks one open
    # document per export unless every exporter closes it afterwards.
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)

    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 1 / UNITS_PER_METER[unit])

    # Get the tools for handling shapes & colors section of the XCAF document.
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    # Auto Naming names shapes and links. Enabling it for glTF's prevents user defined
    # labels from ending up on nodes which is undesirable. Enabling it for STEP files
    # does not seem to affect user labels.
    shape_tool.SetAutoNaming_s(auto_naming)

    # Create a label map to find the appropriate label for all shapes
    label_map: dict[Shape, TDF_Label] = {}

    # Effective display name of each node: an unlabeled node inherits the name
    # of its nearest labeled ancestor so user defined names replace the
    # auto-generated ones (e.g. "SOLID") at every level of the assembly. That
    # label and every node inheriting it are numbered from 0, so every
    # product in the file has a name of its own
    name_map: dict[Shape | None, str] = {}
    numbered: dict[str, int] = {}

    # the labeled nodes whose names unlabeled descendants will inherit
    shared: set[Shape] = set()
    if auto_naming:
        source: dict[Shape | None, Shape | None] = {}
        for node in PreOrderIter(to_export):
            above = source.get(getattr(node, "parent", None))
            source[node] = node if node.label else above
            if not node.label and above is not None:
                shared.add(above)

    def resolve_component_parent_label(label: TDF_Label) -> TDF_Label:
        """Return a label suitable for assembly operations.

        If `label` is a reference/component label, this resolves and returns
        the referred shape label. Otherwise returns `label` unchanged.
        Null labels are returned as-is.
        """
        if label.IsNull():
            return label
        if XCAFDoc_ShapeTool.IsReference_s(label):
            referred = TDF_Label()
            if (
                XCAFDoc_ShapeTool.GetReferredShape_s(label, referred)
                and not referred.IsNull()
            ):
                return referred
        return label

    def set_name_and_color(node: Shape, node_label: TDF_Label, name: str) -> None:
        """Assign label/color metadata for one XDE node label.

        Behavior:
        - Sets `TDataStd_Name` to `name` on the instance label and, when
          applicable, also on the referred shape label so STEP PRODUCT names
          persist.
        - Sets generic color on the instance label and referred label.
        - For leaf `Compound` wrappers (`Part`, `Sketch`, `Curve`), propagates
          color to relevant sub-shapes (solid/face/edge) using appropriate
          XCAF color channels.
        """
        if node_label.IsNull():
            return

        if name:
            TDataStd_Name.Set_s(node_label, TCollection_ExtendedString(name, True))
            if XCAFDoc_ShapeTool.IsReference_s(node_label):
                referred = TDF_Label()
                if (
                    XCAFDoc_ShapeTool.GetReferredShape_s(node_label, referred)
                    and not referred.IsNull()
                ):
                    TDataStd_Name.Set_s(
                        referred, TCollection_ExtendedString(name, True)
                    )

        if node.color is not None:
            node_color_type = XCAFDoc_ColorType.XCAFDoc_ColorGen
            color_tool.SetColor(node_label, node.color.wrapped, node_color_type)

            if XCAFDoc_ShapeTool.IsReference_s(node_label):
                referred = TDF_Label()
                if (
                    XCAFDoc_ShapeTool.GetReferredShape_s(node_label, referred)
                    and not referred.IsNull()
                ):
                    color_tool.SetColor(referred, node.color.wrapped, node_color_type)

            # Sub-shape color handling for leaf Compound wrappers
            if isinstance(node, Compound) and not node.children:
                if isinstance(node, Part):
                    explorer = TopExp_Explorer(node.wrapped, ta.TopAbs_SOLID)
                    sub_color_type = XCAFDoc_ColorType.XCAFDoc_ColorSurf
                elif isinstance(node, Sketch):
                    explorer = TopExp_Explorer(node.wrapped, ta.TopAbs_FACE)
                    sub_color_type = XCAFDoc_ColorType.XCAFDoc_ColorSurf
                elif isinstance(node, Curve):
                    explorer = TopExp_Explorer(node.wrapped, ta.TopAbs_EDGE)
                    sub_color_type = XCAFDoc_ColorType.XCAFDoc_ColorCurv
                else:
                    explorer = TopExp_Explorer()
                    sub_color_type = XCAFDoc_ColorType.XCAFDoc_ColorGen

                shape_label_for_sub = resolve_component_parent_label(node_label)
                while explorer.More():
                    sub_node = explorer.Current()
                    sub_label = shape_tool.AddSubShape(shape_label_for_sub, sub_node)
                    if not sub_label.IsNull():
                        color_tool.SetColor(
                            sub_label, node.color.wrapped, sub_color_type
                        )
                    explorer.Next()

    # Single preorder pass: parent labels are created before children, so we can
    # build label_map and assign metadata without a second traversal.
    for node in PreOrderIter(to_export):
        if node.is_empty:
            continue

        # Don't set parent if node is the one getting exported.
        parent = None if node is to_export else getattr(node, "parent", None)
        if parent is None:
            node_label = shape_tool.AddShape(node.wrapped, False)
        else:
            parent_label = label_map.get(parent, TDF_Label())
            parent_label = resolve_component_parent_label(parent_label)
            if parent_label.IsNull():
                continue
            node_label = shape_tool.AddComponent(parent_label, node.wrapped)

        if node_label.IsNull():
            continue

        label_map[node] = node_label
        inherited = name_map.get(parent, "")
        name_map[node] = node.label or inherited
        node_name = node.label
        if auto_naming and (node in shared or (not node_name and inherited)):
            index = numbered.get(name_map[node], 0)
            node_name = f"{name_map[node]}-{index}"
            numbered[name_map[node]] = index + 1
        if node_name or node.color is not None:
            set_name_and_color(node, node_label, node_name)

    shape_tool.UpdateAssemblies()

    # print(f"Is document valid: {XCAFDoc_DocumentTool.IsXCAFDocument_s(doc)}")

    return doc


def _nothing_to_export(shape: Shape) -> None:
    """An empty shape has nothing to write; say so instead of writing a bad file"""
    if shape.is_empty:
        raise ValueError(
            f"There is nothing to export: the {type(shape).__name__} is empty"
        )


def export_brep(
    to_export: Shape,
    file_path: PathLike | str | bytes | BytesIO | BinaryIO,
    binary: bool = False,
    version: int | None = None,
    triangulation: bool = True,
) -> bool:
    """Export this shape to a BREP file

    Older OCCT versions can't read newer BREP format versions; ``version=1`` with
    ``triangulation=False`` writes a file that any OCCT version can read.

    Args:
        to_export (Shape): object or assembly
        file_path: Union[PathLike, str, bytes, BytesIO]: brep file path or memory buffer
        binary (bool, optional): write the binary BREP format instead of text.
            Defaults to False.
        version (int, optional): BREP format version, 1 to 3 for text and 1 to 4
            for binary. Defaults to None, the latest version.
        triangulation (bool, optional): include the shape's triangulation.
            Defaults to True.

    Raises:
        ValueError: unsupported format version

    Returns:
        bool: write status
    """
    _nothing_to_export(to_export)
    if isinstance(file_path, (PathLike | str | bytes)):
        file_path = fsdecode(file_path)
    else:
        file_path = cast(BytesIO, file_path)

    writer, versions = (
        (BinTools, BinTools_FormatVersion)
        if binary
        else (BRepTools, TopTools_FormatVersion)
    )
    suffix = "CURRENT" if version is None else f"VERSION_{version}"
    format_version = versions.__members__.get(f"{versions.__name__}_{suffix}")
    if format_version is None:
        kind = "binary" if binary else "text"
        raise ValueError(f"Unsupported {kind} BREP version {version}")
    return_value = writer.Write_s(
        to_export.wrapped, file_path, triangulation, False, format_version
    )

    return True if return_value is None else return_value


def export_gltf(
    to_export: Shape,
    file_path: PathLike | str | bytes,
    unit: Unit = Unit.MM,
    binary: bool = False,
    linear_deflection: float = 0.001,
    angular_deflection: float = 0.1,
) -> bool:
    """export_gltf

    The glTF (GL Transmission Format) specification primarily focuses on the efficient
    transmission and loading of 3D models as a compact, binary format that is directly
    renderable by graphics APIs like WebGL, OpenGL, and Vulkan. It's designed to store
    detailed 3D model data, including meshes (vertices, normals, textures, etc.),
    animations, materials, and scene hierarchy, among other aspects.

    UV texture coordinates from the underlying surface parameterization are
    included in the glTF output as ``TEXCOORD_0`` accessors. These UVs support
    texture mapping, UV baking, and displacement-map workflows such as
    `BumpMesh <https://bumpmesh.com/>`_.

    Args:
        to_export (Shape): object or assembly
        file_path (Union[PathLike, str, bytes]): glTF file path
        unit (Unit, optional): shape units. Defaults to Unit.MM.
        binary (bool, optional): output format. Defaults to False.
        linear_deflection (float, optional): A linear deflection setting which limits
            the distance between a curve and its tessellation. Setting this value too
            low will result in large meshes that can consume computing resources. Setting
            the value too high can result in meshes with a level of detail that is too
            low. The default is a good starting point for a range of cases.
            Defaults to 1e-3.
        angular_deflection (float, optional): Angular deflection setting which limits
            the angle between subsequent segments in a polyline. Defaults to 0.1.

    Raises:
        RuntimeError: Failed to write glTF file

    Returns:
        bool: write status
    """
    _nothing_to_export(to_export)

    # Map from OCCT's right-handed +Z up coordinate system to glTF's right-handed +Y
    # up coordinate system
    # https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units
    original_location = to_export.location
    to_export.location *= Location((0, 0, 0), (1, 0, 0), -90)

    # Tessellate the object(s)
    node: Shape
    for node in PreOrderIter(to_export):
        if not node.is_empty:
            node.mesh(linear_deflection, angular_deflection)

    # Create the XCAF document
    doc: TDocStd_Document = _create_xde(to_export, unit, auto_naming=False)

    # Write the glTF file
    writer = RWGltf_CafWriter(
        theFile=TCollection_AsciiString(fsdecode(file_path)), theIsBinary=binary
    )
    writer.SetParallel(True)
    # Textures need UVs in the gltf export
    writer.SetForcedUVExport(True)
    index_map = IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString()
    progress = Message_ProgressRange()

    messenger = Message.DefaultMessenger_s()
    for printer in messenger.Printers():
        printer.SetTraceLevel(Message_Gravity.Message_Fail)

    status = writer.Perform(doc, index_map, progress)

    # Reset tessellation (removes all triangulations including any UV changes)
    BRepTools.Clean_s(to_export.wrapped)

    # Reset original orientation
    to_export.location = original_location

    if not status:
        return status

    # Post-process: inject full PBR materials where nodes have a Material.pbr and
    # normalize the GLTF UVs if material's normalize_uvs is True to
    # ensure different parts of a shape get the same texture pattern size
    # independent of the actual surface size
    node_pbrs: dict[int, PbrProperties] = {}
    for i, node in enumerate(PreOrderIter(to_export)):
        if (
            node.material is not None
            and hasattr(node.material, "pbr")
            and isinstance(node.material.pbr, PbrProperties)
        ):
            pbr = node.material.pbr
            if pbr is not None:
                node_pbrs[i] = pbr

    if node_pbrs:
        file_str = fsdecode(file_path)
        inject_materials(file_str, node_pbrs)

    return status


def export_step(
    to_export: Shape,
    file_path: PathLike | str | bytes | BytesIO | BinaryIO,
    unit: Unit = Unit.MM,
    write_pcurves: bool = True,
    precision_mode: PrecisionMode = PrecisionMode.AVERAGE,
    *,  # Too many positional arguments
    timestamp: str | datetime | None = None,
) -> bool:
    """export_step

    Export a build123d Shape or assembly with color and label attributes.
    Note that if the color of a node in an assembly isn't set, it will be
    assigned the color of its nearest ancestor. Likewise, a node without a
    label is exported under the name of its nearest labeled ancestor, and
    that label and every node inheriting it are numbered from 0 - "box-0",
    "box-1" - so that every product in the file has a name of its own.

    Args:
        to_export (Shape): object or assembly
        file_path (Union[PathLike, str, bytes, BytesIO]): step file path
        unit (Unit, optional): shape units. Defaults to Unit.MM.
        write_pcurves (bool, optional): write parametric curves to the STEP file.
            Defaults to True.
        precision_mode (PrecisionMode, optional): geometric data precision.
            Defaults to PrecisionMode.AVERAGE.

    Raises:
        RuntimeError: Unknown Compound type

    Returns:
        bool: success
    """
    _nothing_to_export(to_export)

    # Create the XCAF document
    doc = _create_xde(to_export, unit, auto_naming=True)

    # Disable writing OCCT info to console
    messenger = Message.DefaultMessenger_s()
    for printer in messenger.Printers():
        printer.SetTraceLevel(Message_Gravity.Message_Fail)

    session = XSControl_WorkSession()
    writer = STEPCAFControl_Writer(session, False)
    writer.SetColorMode(True)
    writer.SetLayerMode(True)
    writer.SetNameMode(True)

    header = APIHeaderSection_MakeHeader(writer.Writer().Model())

    if not header.IsDone():  # As in OCCT 7.9.x
        # Create an empty consistent header, i.e. IsDone() return True
        header = APIHeaderSection_MakeHeader(0)
        header.Apply(writer.Writer().Model())

    if to_export.label:
        header.SetName(TCollection_HAsciiString(to_export.label))
    if timestamp is not None:
        if isinstance(timestamp, datetime):
            header.SetTimeStamp(TCollection_HAsciiString(timestamp.isoformat()))
        else:
            header.SetTimeStamp(TCollection_HAsciiString(timestamp))
    # consider using e.g. the non *Value versions instead
    # header.SetAuthorValue(1, TCollection_HAsciiString("Volker"));
    # header.SetOrganizationValue(1, TCollection_HAsciiString("myCompanyName"));
    header.SetOriginatingSystem(TCollection_HAsciiString("build123d"))
    # header.SetDescriptionValue(1, TCollection_HAsciiString("myApplication Model"));

    STEPCAFControl_Controller.Init_s()
    STEPControl_Controller.Init_s()
    IGESControl_Controller.Init_s()
    Interface_Static.SetIVal_s("write.surfacecurve.mode", int(write_pcurves))
    Interface_Static.SetIVal_s("write.precision.mode", precision_mode.value)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)

    if isinstance(file_path, (PathLike, str, bytes)):
        status = writer.Write(fsdecode(file_path))
    else:
        # need to cast for type checker because BinaryIO is OK but OCP doesn't know
        status = writer.WriteStream(cast(BytesIO, file_path))

    success = status == IFSelect_ReturnStatus.IFSelect_RetDone
    if not success:
        raise RuntimeError("Failed to write STEP file")

    return success


def export_stl(
    to_export: Shape,
    file_path: PathLike | str | bytes,
    tolerance: float = 1e-3,
    angular_tolerance: float = 0.1,
    ascii_format: bool = False,
) -> bool:
    """Export STL

    Exports a shape to a specified STL file.

    Args:
        to_export (Shape): object or assembly
        file_path (Union[PathLike, str, bytes]): The path and file name to write the STL output to.
        tolerance (float, optional): A linear deflection setting which limits the distance
            between a curve and its tessellation. Setting this value too low will result in
            large meshes that can consume computing resources. Setting the value too high can
            result in meshes with a level of detail that is too low. The default is a good
            starting point for a range of cases. Defaults to 1e-3.
        angular_tolerance (float, optional): Angular deflection setting which limits the angle
            between subsequent segments in a polyline. Defaults to 0.1.
        ascii_format (bool, optional): Export the file as ASCII (True) or binary (False)
            STL format. Defaults to False (binary).

    Returns:
        bool: Success

    Raises:
        FileNotFoundError: The destination directory does not exist.
    """
    _nothing_to_export(to_export)
    output_path = Path(fsdecode(file_path))
    if not output_path.parent.is_dir():
        raise FileNotFoundError(output_path.parent)

    mesh = BRepMesh_IncrementalMesh(
        to_export.wrapped, tolerance, True, angular_tolerance, True
    )
    mesh.Perform()

    writer = StlAPI_Writer()

    writer.ASCIIMode = ascii_format
    return writer.Write(to_export.wrapped, str(output_path))


def export_obj(
    to_export: Shape,
    file_path: PathLike | str | bytes | BytesIO | BinaryIO,
    linear_deflection: float = 0.001,
    angular_deflection: float = 0.1,
    include_uvs: bool = True,
    atlas_packing: bool = True,
    atlas_gutter: float = 0.0,
) -> bool:
    """export_obj

    Exports a shape to Wavefront OBJ format.  When *include_uvs* is True
    (default), per-vertex UV texture coordinates are extracted from
    OpenCASCADE's surface parameterization and written as ``vt`` records.
    These UVs enable texture mapping, UV baking, and displacement-map
    workflows such as `BumpMesh <https://bumpmesh.com/>`_ which can add
    displacement heightmap tessellations to 3D-printable meshes.

    When *atlas_packing* is True (default), all face UVs are packed into a
    single texture atlas preserving each face's physical aspect ratio.

    Args:
        to_export (Shape): object or assembly
        file_path (Union[PathLike, str, bytes, BytesIO, BinaryIO]): OBJ file
            path or writable binary stream (for pyodide/WASM environments).
        linear_deflection (float, optional): Tessellation linear deflection.
            Defaults to 1e-3.
        angular_deflection (float, optional): Tessellation angular deflection.
            Defaults to 0.1.
        include_uvs (bool, optional): Include UV texture coordinates.
            Defaults to True.
        atlas_packing (bool, optional): Pack per-face UVs into a single atlas.
            Only used when *include_uvs* is True. Defaults to True.
        atlas_gutter (float, optional): Normalized empty margin around each UV
            island in a packed atlas. Only used when *atlas_packing* is True.
            Defaults to 0.0.

    Returns:
        bool: success
    """
    _nothing_to_export(to_export)
    if include_uvs:
        vertices, triangles, normals, uvs = to_export.tessellate_with_uvs(
            linear_deflection,
            angular_deflection,
            atlas_packing=atlas_packing,
            atlas_gutter=atlas_gutter,
        )
    else:
        vertices, triangles = to_export.tessellate(
            linear_deflection, angular_deflection
        )
        normals = []
        uvs = []

    lines: list[str] = ["# Wavefront OBJ exported by build123d\n"]

    for v in vertices:
        lines.append(f"v {v.X:.9g} {v.Y:.9g} {v.Z:.9g}\n")

    if include_uvs:
        for u_tex, v_tex in uvs:
            lines.append(f"vt {u_tex:.9g} {v_tex:.9g}\n")

        for n in normals:
            lines.append(f"vn {n.X:.9g} {n.Y:.9g} {n.Z:.9g}\n")

        for i0, i1, i2 in triangles:
            a, b, c = i0 + 1, i1 + 1, i2 + 1
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    else:
        for i0, i1, i2 in triangles:
            a, b, c = i0 + 1, i1 + 1, i2 + 1
            lines.append(f"f {a} {b} {c}\n")

    content = "".join(lines)

    if isinstance(file_path, (PathLike, str, bytes)):
        with open(fsdecode(file_path), "w", encoding="utf-8") as f:
            f.write(content)
    else:
        # BytesIO / BinaryIO stream
        file_path.write(content.encode("utf-8"))

    return True


# The color of shapes without one in export_png, as in OCP CAD Viewer
DEFAULT_PNG_COLOR = Color(0xE8B024)


@cache
def _window_class():  # pragma: win32 cover
    """Windows registers a window class once per process. None selects the
    default window procedure"""
    # pylint: disable=import-outside-toplevel
    from OCP.WNT import WNT_WClass

    cs_vredraw, cs_hredraw = 0x0001, 0x0002
    return WNT_WClass(
        TCollection_AsciiString("build123d"), None, cs_vredraw | cs_hredraw
    )


@cache
def _start_ns_application() -> None:  # pragma: darwin cover
    """A Cocoa window needs the application object, which Python doesn't make;
    an application that has one already keeps it"""
    objc = ctypes.cdll.LoadLibrary(
        ctypes.util.find_library("objc") or "/usr/lib/libobjc.dylib"
    )
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.objc_msgSend(
        objc.objc_getClass(b"NSApplication"),
        objc.sel_registerName(b"sharedApplication"),
    )


def _offscreen_window(display: Aspect_DisplayConnection) -> Aspect_Window:
    """A window for OpenGL that is never shown

    OpenGL needs a window to draw for, but ``V3d_View.ToPixMap`` draws into a
    buffer of its own of any size, so the window stays hidden and small, as in
    OpenCascade's own offscreen viewer (``vinit -virtual``).
    """
    # The window classes exist only in the OCP build for their own system
    # pylint: disable=import-outside-toplevel
    window: Aspect_Window
    if sys.platform == "win32":  # pragma: win32 cover
        from OCP.WNT import WNT_Window

        ws_popup = 0x80000000
        window = WNT_Window("build123d", _window_class(), ws_popup, 0, 0, 64, 64)
    elif sys.platform == "darwin":  # pragma: darwin cover
        # A window without a native view (Aspect_NeutralWindow) only works with
        # a graphics device, not with Apple's software renderer
        from OCP.Cocoa import Cocoa_Window

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("On macOS, export_png works only in the main thread")
        _start_ns_application()
        window = Cocoa_Window("build123d", 0, 0, 64, 64)
    else:  # pragma: linux cover
        from OCP.Xw import Xw_Window

        window = Xw_Window(display, "build123d", 0, 0, 64, 64)
    window.SetVirtual(True)
    return window


@cache
def _accelerated_opengl() -> bool:  # pragma: darwin cover
    """Whether this Mac has a graphics device for OpenGL; a virtual machine may
    have none, and then only Apple's software renderer draws"""
    opengl = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    )
    accelerated = 73  # kCGLPFAAccelerated
    attributes = (ctypes.c_int * 2)(accelerated, 0)
    pixel_format, count = ctypes.c_void_p(None), ctypes.c_int(0)
    opengl.CGLChoosePixelFormat(
        attributes, ctypes.byref(pixel_format), ctypes.byref(count)
    )
    if not pixel_format.value:
        return False
    opengl.CGLReleasePixelFormat(pixel_format)
    return True


def _with_dpi(png: bytes, dpi: float) -> bytes:
    """png with its resolution (pHYs) set to dpi, so that it has a size on
    paper, as an SVG has. The encoder may have written one already, and a PNG
    holds at most one"""
    per_metre = round(dpi / 0.0254)
    data = b"pHYs" + struct.pack(">IIB", per_metre, per_metre, 1)
    resolution = struct.pack(">I", 9) + data + struct.pack(">I", zlib.crc32(data))
    chunks, position = [png[:8]], 8
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        kind, end = png[position + 4 : position + 8], position + 12 + length
        if kind != b"pHYs":
            chunks.append(png[position:end])
        if kind == b"IHDR":
            chunks.append(resolution)
        position = end
    return b"".join(chunks)


def _shaded_image(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    to_export: Shape,
    frame: gp_Ax2,
    corner: tuple[float, float],
    pixels: float,
    size: tuple[int, int],
    background: ColorLike,
    line_color: ColorLike | None,
    line_width: float,
) -> Image_AlienPixMap:
    """Draw a shaded picture of to_export as seen in a drawing view's frame,
    with the picture's top left corner at corner in the frame's x and y, and
    pixels pixels per model unit; see export_png"""
    try:
        display = Aspect_DisplayConnection()
    except Aspect_DisplayConnectionDefinitionError as exc:
        raise RuntimeError(
            "export_png needs a display to draw with OpenGL. On Linux without "
            "one, run under a virtual X server, such as with xvfb-run"
        ) from exc
    driver = OpenGl_GraphicDriver(display, False)
    if sys.platform == "darwin":  # pragma: darwin cover
        # Chosen before any view is made: OpenCascade crashes when it removes
        # a view whose OpenGL context could not be made
        driver.ChangeOptions().contextNoAccel = not _accelerated_opengl()
    viewer = V3d_Viewer(driver)
    viewer.SetDefaultLights()
    viewer.SetLightOn()
    context = AIS_InteractiveContext(viewer)
    view = viewer.CreateView()
    view.SetWindow(_offscreen_window(display))
    view.SetBackgroundColor(Color(background).wrapped.GetRGB())
    view.Camera().SetProjectionType(
        Graphic3d_Camera.Projection_e.Projection_Orthographic
    )
    view.ChangeRenderingParams().NbMsaaSamples = 8

    # OpenCascade's default material is shiny: lit from the camera, a face that
    # looks straight at it would come out white
    material = Graphic3d_MaterialAspect(
        Graphic3d_NameOfMaterial.Graphic3d_NameOfMaterial_Plastered
    )
    edges = (
        None
        if line_color is None
        else Prs3d_LineAspect(
            Color(line_color).wrapped.GetRGB(),
            Aspect_TypeOfLine.Aspect_TOL_SOLID,
            line_width,
        )
    )

    # Each leaf at its place in to_export, which is placed as its bounding box
    # and export_step place it, whatever assembly it may be part of
    nodes: list[tuple[Shape, Location]] = [(to_export, Location())]
    while nodes:
        node, location = nodes.pop()
        if node.is_empty:  # nothing to draw, and no location
            continue
        location = location * node.location
        if node.children:
            nodes.extend((child, location) for child in node.children)
            continue
        presentation = AIS_Shape(node.wrapped.Located(location.wrapped))
        presentation.SetMaterial(material)
        color = DEFAULT_PNG_COLOR if node.color is None else node.color
        presentation.SetColor(color.wrapped.GetRGB())
        presentation.SetTransparency(1 - color.wrapped.Alpha())
        if edges is not None:
            presentation.Attributes().SetFaceBoundaryDraw(True)
            presentation.Attributes().SetFaceBoundaryAspect(edges)
        context.Display(presentation, AIS_DisplayMode.AIS_Shaded.value, -1, False)

    # The camera looks along the frame's z axis, with its y axis up, at the
    # middle of the picture, from outside the shape
    x_dir, y_dir, z_dir = (
        Vector(direction)
        for direction in (frame.XDirection(), frame.YDirection(), frame.Direction())
    )
    middle = (
        Vector(frame.Location())
        + x_dir * (corner[0] + size[0] / pixels / 2)
        + y_dir * (corner[1] - size[1] / pixels / 2)
    )
    box = to_export.bounding_box()
    eye = middle + z_dir * (box.diagonal + (box.center() - middle).length + 1)
    camera = view.Camera()
    camera.SetEyeAndCenter(eye.to_pnt(), middle.to_pnt())
    camera.SetUp(frame.YDirection())
    # Graphic3d_Camera's scale is the view's shorter side, in model units
    camera.SetScale(min(size) / pixels)
    image = Image_AlienPixMap()
    if not view.ToPixMap(image, *size):
        raise RuntimeError("OpenGL could not draw the picture")
    return image


def export_png(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    to_export: Shape,
    file_path: PathLike | str | bytes,
    viewpoint: VectorLike | None = None,
    up: VectorLike = (0, 0, 1),
    look_at: VectorLike | None = None,
    unit: Unit = Unit.MM,
    scale: float = 1,
    margin: float = 0,
    fit_to_stroke: bool = True,
    line_color: ColorLike | None = "black",
    line_weight: float = Export2D.DEFAULT_LINE_WEIGHT,
    dpi: float = 96,
    size: tuple[int, int] | None = None,
    background: ColorLike = "white",
) -> bool:
    """export_png

    Write a shaded picture of a shape or assembly to a PNG file, drawn by
    OpenCascade's own renderer with OpenGL and nothing shown on screen. Each
    part of an assembly is drawn at its place in the assembly, in its own color
    or the one it inherits, with its alpha as transparency. Its faces are matte,
    so a face that looks straight at the camera keeps its color.

    The picture is the page :class:`~exporters.ExportSVG` writes for the
    shape's :meth:`~topology.Mixin3D.drawing_view` with the same arguments,
    drawn at dpi: the same view, scale, margin and size on paper, with the
    lines on the picture's edges. Given a size instead, the page is scaled to
    fit in it.

    OpenGL draws only with a display: on Linux without one, run under a virtual
    X server, such as with ``xvfb-run``. On macOS, call it from the main thread.
    Without a graphics card, OpenGL falls back to a software renderer: Mesa's
    on Linux and Apple's on macOS draw the same picture, while Microsoft's
    OpenGL 1.1 on Windows draws the same shapes with simpler shading and no
    antialiasing.

    Args:
        to_export (Shape): object or assembly
        file_path (Union[PathLike, str, bytes]): PNG file path
        viewpoint (VectorLike, optional): where the view is taken from, as in
            drawing_view. Defaults to None, the point (1, -1, 1) from look_at.
        up (VectorLike, optional): direction of the picture's y axis, as in
            drawing_view. Defaults to (0, 0, 1).
        look_at (VectorLike, optional): point the view is centred on, as in
            drawing_view. Defaults to None (center of shape).
        unit (Unit, optional): unit of the page, as in ExportSVG.
            Defaults to Unit.MM.
        scale (float, optional): page units per model unit, as in ExportSVG.
            Defaults to 1.
        margin (float, optional): empty border around the shape, in page
            units, as in ExportSVG. Defaults to 0.
        fit_to_stroke (bool, optional): whether the border also fits half the
            line weight, as in ExportSVG. Defaults to True.
        line_color (ColorLike, optional): color of the edges between faces, or
            None to leave them out. Defaults to "black".
        line_weight (float, optional): width of the edges in millimetres, as in
            ExportSVG. Defaults to Export2D.DEFAULT_LINE_WEIGHT.
        dpi (float, optional): pixels per inch of the page, which the PNG
            records. Defaults to 96, at which browsers show an SVG.
        size (tuple[int, int], optional): picture width and height in pixels,
            the page scaled to fit in them, in their middle; scale is then not
            used. Defaults to None, the page's size at dpi.
        background (ColorLike, optional): background color. Defaults to "white".

    Raises:
        ValueError: viewpoint equals look_at, up is parallel to the view, scale
            or dpi isn't positive, or size has no room inside the margin
        RuntimeError: no display, or the picture could not be drawn
        FileNotFoundError: The destination directory does not exist.

    Returns:
        bool: success
    """
    _nothing_to_export(to_export)
    if scale <= 0 or dpi <= 0:
        raise ValueError(f"scale and dpi must be positive, not {scale} and {dpi}")
    if size is not None and min(size) < 1:
        raise ValueError(f"size must be at least one pixel each way, not {size}")
    look_at = _drawing_look_at(to_export, look_at)
    if viewpoint is None:
        viewpoint = look_at + (1, -1, 1)
    direction = Vector(viewpoint) - look_at
    if direction.length < TOLERANCE:
        raise ValueError("viewpoint and look_at must be different points")
    if direction.normalized().cross(Vector(up).normalized()).length < TOLERANCE:
        raise ValueError("up must not be parallel to the direction of view")
    frame = _drawing_frame(viewpoint, up, look_at)

    # The shape's bounds in drawing_view's coordinates, as ExportSVG gets them
    # from its lines
    to_frame = gp_Trsf()
    to_frame.SetTransformation(gp_Ax3(frame))
    bounds = BoundBox(to_export.wrapped.Moved(TopLoc_Location(to_frame)))
    stroke = line_weight if fit_to_stroke else 0
    pixels_per_page_unit = dpi / unit_conversion_scale(Unit.IN, unit)
    if size is not None:
        # The largest scale whose page fits in size
        room = [side / pixels_per_page_unit - 2 * margin - stroke for side in size]
        if min(room) <= 0:
            raise ValueError(f"a margin of {margin} leaves no room in {size} pixels")
        extents = (bounds.size.X, bounds.size.Y)
        scale = min((r / e for r, e in zip(room, extents) if e > 0), default=scale)
    left, top, width, height = _page(bounds, scale, margin, stroke)
    pixels = scale * pixels_per_page_unit
    if size is None:
        size = (
            max(1, ceil(round(width * pixels, 6))),
            max(1, ceil(round(height * pixels, 6))),
        )
        corner = (left, top)
    else:
        corner = (
            left + (width - size[0] / pixels) / 2,
            top - (height - size[1] / pixels) / 2,
        )

    image = _shaded_image(
        to_export,
        frame,
        corner,
        pixels,
        size,
        background,
        line_color,
        # line_weight is in millimetres, as in ExportSVG; thinner than a pixel,
        # a line would fade, so it is drawn a pixel wide, as a hairline
        max(1.0, line_weight * dpi / 25.4),
    )
    # OpenCascade picks the format from a file name's extension, so the PNG is
    # made in memory and then written, whatever the file is called
    png = BytesIO()
    if not image.Save(png, TCollection_AsciiString(".png")):
        raise RuntimeError("OpenCascade could not encode the picture as PNG")
    with open(fsdecode(file_path), "wb") as file:
        file.write(_with_dpi(png.getvalue(), dpi))
    return True


def export_to_pcbway(
    to_export: Shape,
    unit: Unit = Unit.MM,
    write_pcurves: bool = True,
    precision_mode: PrecisionMode = PrecisionMode.AVERAGE,
) -> str:
    """Export a shape to PCBWay for quoting.

    This function writes ``to_export`` to a temporary STEP file, uploads that file
    to PCBWay's external web service, opens the returned pricing page in the
    default browser, and returns the pricing page URL.

    Args:
        to_export (Shape): object or assembly
        unit (Unit, optional): shape units. Defaults to Unit.MM.
        write_pcurves (bool, optional): write parametric curves to the STEP file.
            Defaults to True.
        precision_mode (PrecisionMode, optional): geometric data precision.
            Defaults to PrecisionMode.AVERAGE.

    Returns:
        str: URL of the pricing page
    """
    # PCBWay specific URLs for build123d uploads
    upload_url = "https://www.pcbway.com/common/Build123dUpFile"
    redirect_url = ""

    # Export the part to a temp file
    step_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            step_file = Path(tmp.name)

        # Re-open the tmp file to help with Windows compatibility
        export_step(
            to_export,
            step_file,
            unit=unit,
            write_pcurves=write_pcurves,
            precision_mode=precision_mode,
        )

        # Transfer the step file
        with open(step_file, "rb") as f:
            response = requests.post(
                upload_url,
                files={"file": (step_file.name, f, "application/step")},
                timeout=(10, 120),
            )
        response.raise_for_status()

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"PCBWay returned a non-JSON response: {response.text[:500]}"
            ) from exc
        redirect_url = result.get("redirect")
        if result.get("state") != "SUCCESS" or not isinstance(redirect_url, str):
            raise RuntimeError(
                f"PCBWay upload failed or returned no redirect: {result}"
            )

    # Always remove the tmp file
    finally:
        if step_file is not None:
            step_file.unlink(missing_ok=True)

    if not webbrowser.open(redirect_url, new=2):
        warnings.warn(f"webbrowser failed to open on pricing page {redirect_url}")

    return redirect_url
