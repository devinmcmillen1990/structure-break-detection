import numpy as np
from scipy.spatial import ConvexHull

def icosphere(subdivisions=2):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0],[ 1,  t,  0],[-1, -t,  0],[ 1, -t,  0],
        [ 0, -1,  t],[ 0,  1,  t],[ 0, -1, -t],[ 0,  1, -t],
        [ t,  0, -1],[ t,  0,  1],[-t,  0, -1],[-t,  0,  1]
    ], dtype=float)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=int)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    def midpoint(a, b, cache, vlist):
        key = tuple(sorted((a, b)))
        if key in cache:
            return cache[key]
        m = (vlist[a] + vlist[b]) * 0.5
        m /= np.linalg.norm(m)
        vlist.append(m)
        idx = len(vlist) - 1
        cache[key] = idx
        return idx

    vlist = [v for v in verts]
    for _ in range(subdivisions):
        new_faces = []
        mid_cache = {}
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b, mid_cache, vlist)
            bc = midpoint(b, c, mid_cache, vlist)
            ca = midpoint(c, a, mid_cache, vlist)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = np.array(new_faces, dtype=int)
    verts = np.array(vlist, dtype=float)
    return verts, faces

def plot_mesh(verts, faces, title="Mesh"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    tri = Poly3DCollection(verts[faces], alpha=0.9, linewidths=0.2, edgecolor='k')
    ax.add_collection3d(tri)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout(); plt.show()

def star_body_from_points(points, quantile=0.90, ico_subdiv=3):
    pts = np.asarray(points, dtype=np.float32)
    dirs, faces = icosphere(subdivisions=ico_subdiv)
    proj = pts @ dirs.T                      # (N, M)
    r = np.quantile(proj, quantile, axis=0)  # robust radius
    verts = dirs * r[:, None]
    return verts.astype(np.float32), faces

def rounded_hull_support_mesh(points, r=0.08, ico_subdiv=3):
    """
    Memory-safe approximation of (convex hull ⊕ sphere(r)) using support sampling.
    For direction u: support h_K(u)=max_{x in K} u·x, so h_{K⊕Br}(u)=h_K(u)+r.
    We sample directions on an icosphere and build the mesh with that topology.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 4:
        # Not enough points for a hull; fall back to star body
        return star_body_from_points(pts, quantile=1.0, ico_subdiv=ico_subdiv)

    # Optionally reduce to hull vertices (keeps support exact while shrinking N)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    dirs, faces = icosphere(subdivisions=ico_subdiv)   # unit directions (M,3)
    # support per direction
    proj = hull_pts @ dirs.T                           # (Nh, M)
    h = proj.max(axis=0)                               # (M,)
    rdir = h + r                                       # Minkowski add with ball radius r
    verts = dirs * rdir[:, None]
    return verts.astype(np.float32), faces
