import unittest
import networkx as nx
from utils.graph_ops import dag_to_cpdag

class TestCPDAG(unittest.TestCase):
    def test_dag_to_cpdag_v_structure(self):
        # X -> Y <- Z (V-structure, should be preserved)
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Z", "Y")])
        
        cpdag = dag_to_cpdag(dag)
        
        self.assertTrue(cpdag.has_edge("X", "Y"))
        self.assertFalse(cpdag.has_edge("Y", "X"))
        self.assertTrue(cpdag.has_edge("Z", "Y"))
        self.assertFalse(cpdag.has_edge("Y", "Z"))
        
    def test_dag_to_cpdag_fork(self):
        # Y <- X -> Z (Fork, should become undirected Y - X - Z)
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("X", "Z")])
        
        cpdag = dag_to_cpdag(dag)
        
        # Check X-Y
        self.assertTrue(cpdag.has_edge("X", "Y"))
        self.assertTrue(cpdag.has_edge("Y", "X"))
        
        # Check X-Z
        self.assertTrue(cpdag.has_edge("X", "Z"))
        self.assertTrue(cpdag.has_edge("Z", "X"))

    def test_dag_to_cpdag_chain(self):
        # X -> Y -> Z (Chain, should become undirected X - Y - Z)
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])
        
        cpdag = dag_to_cpdag(dag)
        
        # Check X-Y
        self.assertTrue(cpdag.has_edge("X", "Y"))
        self.assertTrue(cpdag.has_edge("Y", "X"))
        
        # Check Y-Z
        self.assertTrue(cpdag.has_edge("Y", "Z"))
        self.assertTrue(cpdag.has_edge("Z", "Y"))

if __name__ == "__main__":
    unittest.main()
