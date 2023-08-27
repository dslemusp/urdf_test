import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

g = nx.DiGraph()


nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


class Rotation(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    rpy: np.ndarray | list[float] = np.array([0.0, 0.0, 0.0])

    @staticmethod
    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    @staticmethod
    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    
    @computed_field
    @property
    def rx(self) -> float:
        return self.rpy[0]
    
    @rx.setter
    def rx(self, value: float):
        self.rpy[0] = value

    @computed_field
    @property
    def ry(self) -> float:
        return self.rpy[1]
    
    @ry.setter
    def ry(self, value: float):
        self.rpy[1] = value

    @computed_field
    @property
    def rz(self) -> float:
        return self.rpy[2]
    
    @rz.setter
    def rz(self, value: float):
        self.rpy[2] = value

    @computed_field
    @property
    def matrix(self) -> np.ndarray:
        return self.rot_z(self.rpy[2]) @ self.rot_y(self.rpy[1]) @ self.rot_x(self.rpy[0])
    
    def __mult__(self, other):
        return self.matrix @ other.matrix


class Vector(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    vector: np.ndarray | list[float] = np.array([0.0, 0.0, 0.0])

    @computed_field
    @property
    def x(self) -> np.ndarray:
        return self.vector[0]
    
    @x.setter
    def x(self, value: float):
        self.vector[0] = value
    
    @computed_field
    @property
    def y(self) -> np.ndarray:
        return self.vector[1]
    
    @y.setter
    def y(self, value: float):
        self.vector[1] = value
    
    @computed_field
    @property
    def z(self) -> np.ndarray:
        return self.vector[2]
    
    @z.setter
    def z(self, value: float):
        self.vector[2] = value
    

    def unit(self):
        return Vector(vector=self.vector / np.linalg.norm(self.vector))

    def __add__(self, other):
        return Vector(vector=self.vector + other.vector)

    def __sub__(self, other):
        return Vector(vector=self.vector - other.vector)


class Transformation(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    R: Rotation = Rotation()
    p: Vector = Vector()
    parent: str = None
    child: str = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.parent == other.parent and self.child == other.child
        else:
            return False

    def __hash__(self):
        return id(self)


pa = Transformation(parent="A", child="B")
pb = Transformation(parent="A", child="B")

print(pa == pb)

poses = [Transformation(parent=source, child=target) for source, target in zip(nodes[:-1], nodes[1:])]

# print([source == target for source,target in zip(nodes[:-1],nodes[1:])])
print([source == target for source, target in zip(poses[:-1], poses[1:])])

[g.add_node(node) for node in poses]
print(g.nodes(data=True))

# [g.add_edge(source,target, T=transformation) for source,target,transformation in zip(nodes[:-1],nodes[1:],[ab, bc])]
# [g.add_edge(target,source, T=transformation) for source,target,transformation in zip(nodes[:-1],nodes[1:],[ab, bc])]
