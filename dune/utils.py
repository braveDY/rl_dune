from typing import Tuple, Optional
import numpy as np


def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
def check_convex_order(points: np.ndarray) -> Tuple[bool, Optional[str]]:
    n = points.shape[1]
    if n < 3:
        return False, None
    
    direction = 0
    for i in range(n):
        o, a, b = points[:, i], points[:, (i+1)%n], points[:, (i+2)%n]
        cp = cross(o, a, b)
        
        if cp != 0:
            if direction == 0:
                direction = 1 if cp > 0 else -1
            elif (cp > 0 and direction < 0) or (cp < 0 and direction > 0):
                return False, None
    
    return True, "CCW" if direction > 0 else "CW"
def convex_vertex_to_ineq(vertices: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    is_convex, order = check_convex_order(vertices)
    if not is_convex:
        print("Warning: Polygon is not convex")
        return None, None

    if order == "CW":
        vertices = np.hstack([vertices[:, :1], vertices[:, 1:][:, ::-1]])

    n_edges = vertices.shape[1]
    G = np.zeros((n_edges, 2))
    h = np.zeros((n_edges, 1))

    for i in range(n_edges):
        p1 = vertices[:, i]
        p2 = vertices[:, (i+1)%n_edges]
        diff = p2 - p1
        a, b = diff[1], -diff[0]
        c = a * p1[0] + b * p1[1]
        G[i] = [a, b]
        h[i] = c

    return G, h

def diff_ineq(length, width):
    vertices = np.array([
        [-length/2, length/2, length/2, -length/2],
        [-width/2, -width/2, width/2, width/2]
    ])
    return convex_vertex_to_ineq(vertices)


