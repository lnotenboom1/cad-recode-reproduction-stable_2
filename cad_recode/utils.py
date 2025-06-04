# cad_recode/utils.py
import numpy as np
import cadquery as cq
from cadquery import exporters
from itertools import count
import os

# --------------------------------------------------------------------------- #
# Robust tessellation-based surface sampler (debug-friendly)
# --------------------------------------------------------------------------- #
def sample_points_on_shape(shape, n_samples: int = 1024, tol: float = 0.2):
    """
    Uniformly sample `n_samples` points on the surface of *shape*.
    Works with CadQuery ≥ 2.0 by trying every known tessellate signature.
    The returned points are normalized (centered and scaled).
    """
    # Convert Workplane to solid if needed
    if isinstance(shape, cq.Workplane):
        try:
            shape = shape.val()
        except Exception:
            # If workplane contains multiple solids, just use the first one
            shape = shape.objects[0] if shape.objects else shape
    # ---------------------------------------------------------------------- #
    # 1) Tessellate robustly
    call_variants = (
        ("positional",            lambda: shape.tessellate(tol)),               # ≤ 2.0
        ("angular_tolerance=tol", lambda: shape.tessellate(angular_tolerance=tol)),  # 2.1–2.3
        ("tolerance=tol",         lambda: shape.tessellate(tolerance=tol)),    # ≥ 2.4
    )
    verts = faces = None
    for tag, call in call_variants:
        try:
            verts, faces = call()
            break
        except TypeError:
            continue
        except Exception:
            continue
    if verts is None or faces is None:
        raise RuntimeError("CadQuery API change: could not tessellate shape.")

    # Convert verts to float array
    if hasattr(verts[0], "x"):
        verts = [[v.x, v.y, v.z] for v in verts]
    verts = np.asarray(verts, dtype=np.float32)  # (M, 3)
    faces = np.asarray(faces, dtype=np.int64)    # (K, 3)

    # ---------------------------------------------------------------------- #
    # 2) Importance-sample triangles
    tri = verts[faces]  # (K, 3, 3)
    # compute triangle areas
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1
    )
    # sample triangles proportional to area
    total_area = area.sum()
    if total_area <= 0:
        raise RuntimeError("Shape has zero surface area; cannot sample points.")
    pdf = area / total_area
    # randomly choose triangles
    choice = np.random.choice(len(tri), size=n_samples, p=pdf)
    tri = tri[choice]
    # random barycentric coordinates
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    # ensure u+v <= 1 for points inside triangle
    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)
    # sample points as weighted vertices
    pts = u * tri[:, 0] + v * tri[:, 1] + w * tri[:, 2]

    # ---------------------------------------------------------------------- #
    # 3) Normalize point cloud (center and scale to unit radius)
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-9:
        pts = pts / scale

    return pts.astype(np.float32)

# initialize call counter
sample_points_on_shape._counter = 0

def farthest_point_sample(points: np.ndarray, k: int):
    """Classic Farthest Point Sampling (FPS) in NumPy."""
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N == 0 or k <= 0:
        return np.empty((0, 3), dtype=np.float32)
    k = min(k, N)
    sel = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf)
    sel[0] = 0  # start at first point
    for i in range(1, k):
        d = np.linalg.norm(points - points[sel[i-1]], axis=1)
        dist = np.minimum(dist, d)
        sel[i] = dist.argmax()
    return points[sel]

def chamfer_distance(a: np.ndarray, b: np.ndarray):
    """
    L2-squared symmetric Chamfer distance between two point sets (N,3) / (M,3).
    Uses a KD-tree for faster distance queries if SciPy is available.
    """
    try:
        from scipy.spatial import cKDTree
        d_ab = cKDTree(a).query(b)[0] ** 2
        d_ba = cKDTree(b).query(a)[0] ** 2
        return d_ab.mean() + d_ba.mean()
    except ImportError:
        # brute-force fallback (may be slow)
        d_ab = ((b[:, None, :] - a[None, :, :])**2).sum(-1).min(axis=1)
        d_ba = ((a[:, None, :] - b[None, :, :])**2).sum(-1).min(axis=1)
        return d_ab.mean() + d_ba.mean()

def save_point_cloud(points: np.ndarray, filename: str):
    """
    Save a point cloud (N x 3 NumPy array) to an ASCII PLY file.
    """
    points = np.asarray(points, dtype=np.float32)
    assert points.ndim == 2 and points.shape[1] == 3, "Points must be an (N,3) array."
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    N = points.shape[0]
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        # write points
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def edit_distance(a, b):
    """
    Compute the Levenshtein edit distance between sequences a and b.
    Both `a` and `b` can be lists or strings.
    """
    # Convert to lists of tokens if strings
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    len_a, len_b = len(a), len(b)
    # initialize DP matrix
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len_a][len_b]
