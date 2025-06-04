import numpy as np
from utils.helpers import causallearn_to_dag


def test_causallearn_to_dag_simple():
    amat = np.array([[0, 1], [-1, 0]])
    dag = causallearn_to_dag(amat, ['X', 'Y'])
    assert list(dag.edges()) == [('X', 'Y')]

