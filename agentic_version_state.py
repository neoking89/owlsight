import uuid
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import json
import copy

class VersionedState:
    def __init__(self, data: Any, parent: Optional['VersionedState'] = None, 
                 description: str = "", tags: List[str] = None):
        self.id = str(uuid.uuid4())
        self.data = data
        self.parent = parent
        self.children: List['VersionedState'] = []
        self.timestamp = datetime.now().isoformat()
        self.description = description
        self.tags = tags or []
        
    def add_version(self, data: Any, description: str = "", tags: List[str] = None) -> 'VersionedState':
        new_version = VersionedState(data, parent=self, description=description, tags=tags)
        self.children.append(new_version)
        return new_version
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            "id": self.id,
            "data": self.data,
            "timestamp": self.timestamp,
            "description": self.description,
            "tags": self.tags,
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [child.id for child in self.children]
        }
    
    def __str__(self) -> str:
        return f"State({self.id[:8]}): {self.description or self.timestamp}"

class CentralMemory:
    def __init__(self, name: str = "Default Memory", description: str = "Version control system"):
        self.name = name
        self.description = description
        self.root: VersionedState = VersionedState("initial", description="Root state")
        self.current: VersionedState = self.root
        self.states_map: Dict[str, VersionedState] = {self.root.id: self.root}
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at
        
    def update(self, data: Any, description: str = "", tags: List[str] = None) -> VersionedState:
        self.current = self.current.add_version(data, description, tags)
        self.states_map[self.current.id] = self.current
        self.last_modified = datetime.now().isoformat()
        return self.current
    
    def rollback(self) -> Optional[VersionedState]:
        """Roll back to parent state"""
        if self.current.parent:
            self.current = self.current.parent
            self.last_modified = datetime.now().isoformat()
            return self.current
        return None
    
    def forward(self, child_index: int = 0) -> Optional[VersionedState]:
        """Move forward to a child state"""
        if 0 <= child_index < len(self.current.children):
            self.current = self.current.children[child_index]
            self.last_modified = datetime.now().isoformat()
            return self.current
        return None
    
    def goto(self, state_id: str) -> bool:
        """Go to specific state by ID"""
        if state_id in self.states_map:
            self.current = self.states_map[state_id]
            self.last_modified = datetime.now().isoformat()
            return True
        return False
    
    def find_by_tag(self, tag: str) -> List[VersionedState]:
        """Find states by tag"""
        return [state for state in self.states_map.values() if tag in state.tags]
    
    def branch(self, data: Any, description: str = "", tags: List[str] = None) -> VersionedState:
        """Create a new branch from the current state"""
        return self.update(data, description, tags)
    
    def read(self) -> Any:
        """Read current state data"""
        return self.current.data
    
    def count_all_states(self) -> int:
        """Count all states in the version tree"""
        return len(self.states_map)
    
    def count_branch_length(self) -> int:
        """Count length from root to current state"""
        count = 1  # Start with current node
        node = self.current
        while node.parent:
            count += 1
            node = node.parent
        return count
    
    def export_to_json(self, filepath: str) -> None:
        """Export the version tree to JSON"""
        tree_data = {state_id: state.to_dict() for state_id, state in self.states_map.items()}
        with open(filepath, 'w') as f:
            json.dump({
                "name": self.name,
                "description": self.description,
                "created_at": self.created_at,
                "last_modified": self.last_modified,
                "current_id": self.current.id,
                "root_id": self.root.id,
                "states": tree_data
            }, f, indent=2)
    
    def visualize(self, figsize=(10, 8), show_labels=True, highlight_current=True):
        """Visualize the version tree using NetworkX and Matplotlib"""
        G = nx.DiGraph()
        
        # Add all nodes and edges to the graph
        for state_id, state in self.states_map.items():
            # Add node with shortened ID as label
            label = f"{state.id[:6]}: {state.description[:20]}" if state.description else state.id[:8]
            G.add_node(state.id, label=label)
            
            # Add edge from parent to this node
            if state.parent:
                G.add_edge(state.parent.id, state.id)
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, 
                              node_color=['red' if n == self.current.id else 'skyblue' for n in G.nodes])
        
        # Draw all edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15)
        
        # Draw labels if requested
        if show_labels:
            labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f"{self.name} - Version Tree ({len(self.states_map)} states)")
        plt.axis('off')
        return plt
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description} - {self.count_all_states()} states"

# ---- Example usage ----
if __name__ == "__main__":
    # Create a named memory system
    memory = CentralMemory("Project Alpha", "Document versioning system for project documents")
    
    # Add some versions
    memory.update("Version 1 data", "First draft")
    memory.update("Version 2 data", "Added introduction")
    
    # Create a branch
    branch_point = memory.current
    memory.update("Version 3A data", "Option A implementation", tags=["option-a"])
    
    # Go back and create another branch
    memory.goto(branch_point.id)
    memory.update("Version 3B data", "Option B implementation", tags=["option-b"])
    
    # Visualize the version tree
    plt.figure(figsize=(12, 8))
    memory.visualize()
    plt.savefig("version_tree.png")
    plt.show()
    
    # Export to JSON
    memory.export_to_json("project_alpha_versions.json")
    
    print(f"Memory system info: {memory}")
    print(f"Current state: {memory.current}")