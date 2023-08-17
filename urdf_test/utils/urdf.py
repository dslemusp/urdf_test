import os
import pyrender  # Save pyrender import for here for CI
import six
from lxml import etree as ET
from urchin import URDF as URDF_base
from typing_extensions import Any

class URDFLocal(URDF_base):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def load(file_obj: Any, lazy_load_meshes: bool = False) -> "URDFLocal":
        """Load a URDF from a file.

        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        lazy_load_meshes : bool
            If true, meshes will only loaded when requested by a function call.
            This dramatically speeds up loading time for the URDF but may lead
            to unexpected timing mid-program when the meshes have to be loaded

        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, six.string_types):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return URDFLocal._from_xml(node, path, lazy_load_meshes)

    @classmethod
    def _from_xml(cls, node: Any, path: Any, lazy_load_meshes: bool) -> "URDFLocal":
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, lazy_load_meshes)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        return URDFLocal(**kwargs)

    def show(self, cfg: Any | None = None, use_collision: bool=False) -> pyrender.Scene:
        """Visualize the URDF in a given configuration.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        use_collision : bool
            If True, the collision geometry is visualized instead of
            the visual geometry.
        run_in_thread: bool
            asdfsdf
            asf
        """

        if use_collision:
            fk = self.collision_trimesh_fk(cfg=cfg)
        else:
            fk = self.visual_trimesh_fk(cfg=cfg)

        scene = pyrender.Scene()
        nodes = []
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            nodes.append(scene.add(mesh, pose=pose))

        # return pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=run_in_thread, auto_start=False)
        return (scene, nodes)
