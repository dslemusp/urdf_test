import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from typing import Annotated


def _validate_list(v: NDArray[np.float64] | list[float]) -> NDArray:
    if isinstance(v, list):
        return np.array(v)
    return v


class Rotation(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    rpy: NDArray[np.float64] | list[float] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    _rpy_validator = field_validator("rpy")(_validate_list)

    @staticmethod
    def rot_x(angle: float) -> NDArray:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def rot_y(angle: float) -> NDArray:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    @staticmethod
    def rot_z(angle: float) -> NDArray:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    @computed_field  # type: ignore[misc]
    @property
    def rx(self) -> float:
        return self.rpy[0]

    @rx.setter
    def rx(self, value: float) -> None:
        self.rpy[0] = value

    @computed_field  # type: ignore[misc]
    @property
    def ry(self) -> float:
        return self.rpy[1]

    @ry.setter
    def ry(self, value: float) -> None:
        self.rpy[1] = value

    @computed_field  # type: ignore[misc]
    @property
    def rz(self) -> float:
        return self.rpy[2]

    @rz.setter
    def rz(self, value: float) -> None:
        self.rpy[2] = value

    @computed_field  # type: ignore[misc]
    @property
    def matrix(self) -> NDArray:
        return self.rot_z(self.rpy[2]) @ self.rot_y(self.rpy[1]) @ self.rot_x(self.rpy[0])
    
    @matrix.setter
    def matrix(self, matrix: NDArray) -> None:
        """
        Interprets the provided matrix as a 3x3 rotation matrix and returns Euler angles in radians to be applied as
        individual sequential rotations about axes x, y and then z (extrinsic).
        """
        # See: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

        sy = -matrix[2, 0]
        if np.isclose(sy, 1.0):
            rx = np.arctan2(matrix[0, 1], matrix[0, 2])
            ry = 0.5 * np.pi
            rz = 0.0
        elif np.isclose(sy, -1.0):
            rx = np.arctan2(-matrix[0, 1], -matrix[0, 2])
            ry = -0.5 * np.pi
            rz = 0.0
        else:
            cy_inv = 1.0 / np.sqrt(1.0 - sy * sy)  # 1 / cos(ry)
            rx = np.arctan2(matrix[2, 1] * cy_inv, matrix[2, 2] * cy_inv)
            ry = np.arcsin(sy)
            rz = np.arctan2(matrix[1, 0] * cy_inv, matrix[0, 0] * cy_inv)
        
        self.rpy = np.array([rx, ry, rz])


    def __mult__(self, other: "Rotation") -> NDArray:
        return self.matrix @ other.matrix


class Vector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    vector: NDArray[np.float64] | list[float] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    _vector_validator = field_validator("vector")(_validate_list)

    @computed_field  # type: ignore[misc]
    @property
    def x(self) -> NDArray:
        return self.vector[0]

    @x.setter
    def x(self, value: float) -> None:
        self.vector[0] = value

    @computed_field  # type: ignore[misc]
    @property
    def y(self) -> NDArray:
        return self.vector[1]

    @y.setter
    def y(self, value: float) -> None:
        self.vector[1] = value

    @computed_field  # type: ignore[misc]
    @property
    def z(self) -> NDArray:
        return self.vector[2]

    @z.setter
    def z(self, value: float) -> None:
        self.vector[2] = value

    def to_unit(self) -> "Vector":
        return Vector(vector=self.vector / np.linalg.norm(self.vector))  # type: ignore[call-overload, operator]

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(vector=self.vector + other.vector)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(vector=self.vector - other.vector)  # type: ignore[operator]


class Transformation(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    pose: NDArray[np.float64] | list[float] = Field(
        default=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), min_length=6, max_length=6
    )
    parent: str | None = None
    child: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def translation(self) -> Vector:
        return Vector(vector=self.pose[:3])
    
    @translation.setter
    def translation(self, value: Vector) -> None:
        self.pose[:3] = value.vector

    @computed_field  # type: ignore[misc]
    @property
    def rotation(self) -> Rotation:
        return Rotation(rpy=self.pose[3:])
    
    @rotation.setter
    def rotation(self, value: Rotation) -> None:
        self.pose[3:] = value.rpy

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.parent == other.parent and self.child == other.child
        else:
            return False

    def __hash__(self) -> int:
        return id(self)


if __name__ == "__main__":
    g = nx.DiGraph()

    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    pa = Transformation(parent="A", child="B")
    pb = Transformation(parent="A", child="B")

    print(pa == pb)

    poses = [Transformation(parent=source, child=target) for source, target in zip(nodes[:-1], nodes[1:])]

    # print([source == target for source,target in zip(nodes[:-1],nodes[1:])])
    print([source == target for source, target in zip(poses[:-1], poses[1:])])

    [g.add_node(node) for node in poses]
    print(g.nodes(data=True))

    # [
    #     g.add_edge(source, target, T=transformation)
    #     for source, target, transformation in zip(nodes[:-1], nodes[1:], [ab, bc])
    # ]
    # [
    #     g.add_edge(target, source, T=transformation)
    #     for source, target, transformation in zip(nodes[:-1], nodes[1:], [ab, bc])
    # ]
