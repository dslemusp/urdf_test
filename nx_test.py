import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict

g = nx.DiGraph()


nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

class Transformation(BaseModel):
    model_config: ConfigDict = {"arbitrary_types_allowed": True}

    R : np.ndarray = np.eye(3)
    p : np.ndarray = np.zeros(3)
    parent: str = None
    child: str = None

    def __eq__(self, other): 
        if isinstance(other, self.__class__):
            return  self.parent == other.parent and self.child == other.child
        else:
            return False

    def __hash__(self):
        return id(self)
    
pa = Transformation(parent = "A", child = "B")
pb = Transformation(parent = "A", child = "B")

print(pa == pb)

poses = [Transformation(parent=source, child=target) for source,target in zip(nodes[:-1],nodes[1:])]

# print([source == target for source,target in zip(nodes[:-1],nodes[1:])])
print([source == target for source,target in zip(poses[:-1],poses[1:])])

[g.add_node(node) for node in poses]
print(g.nodes(data=True))

# [g.add_edge(source,target, T=transformation) for source,target,transformation in zip(nodes[:-1],nodes[1:],[ab, bc])]
# [g.add_edge(target,source, T=transformation) for source,target,transformation in zip(nodes[:-1],nodes[1:],[ab, bc])]

