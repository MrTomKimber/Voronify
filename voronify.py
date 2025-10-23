from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import linear_sum_assignment


from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate, scale, affine_transform

from matplotlib.path import Path

import numpy as np
import random
from math import sqrt, e, pi, cos, sin, ceil
from itertools import product

def logit(x, l,k,xo):
    return l / ( 1 + (e**(-k*(x-xo))))

# Code used to surround an intended Voronoi pointset with boundary points to avoid
# creation of degenerate edges
def bb_poly(bounds):
    ax,ay,bx,by = bounds
    return Polygon([
        (ax,ay), (ax,by), (bx,by), (bx,ay)
    ])

def poly_radius(poly: Polygon):
    cx, cy = poly.centroid.x,poly.centroid.y 
    print(cx, cy)
    maxd = 0
    for px, py in poly.exterior.coords:
        d = (((px-cx)**2)+((py-cy)**2))**0.5
        if d > maxd:
            maxd = d
    return maxd

def svg_path_to_path_points(strpath):
    path_points=[]
    for t in strpath.split("L"):
        x,y = map(float, t.replace("M","").replace("Z","").strip().split(" "))
        path_points.append((x,y))
    xs,ys = list(zip(*path_points))
    minx, maxx = min(xs), max(xs) 
    miny, maxy = min(ys), max(ys)
    xrng, yrng = maxx-minx, maxy-miny

    return path_points


# Given an aspect-ratio, and a number of shapes, describe the lattice that 
# would be needed to host those shapes:
def define_aspect(number_of_shapes, aspect_ratio):
    x = ceil(sqrt(number_of_shapes * aspect_ratio))
    y = ceil(number_of_shapes / x)
    return x,y

def shapely_to_path(geom):
    """
    Convert a (multi‑)polygon into a Matplotlib Path.
    Works for simple polygons – for complex cases you might
    need to split into individual polygons.
    """
    # shapely.geometry.mapping returns a dict with 'type' & 'coordinates'
    gmap = mapping(geom)
    if gmap['type'] == 'Polygon':
        exteriors = [np.array(gmap['coordinates'][0])]

    verts = exteriors[0]
    codes = [Path.MOVETO] + ([Path.LINETO] * (len(verts) - 2)) + [Path.CLOSEPOLY]
    return Path(verts, codes)

def _boundary_points(outline: Polygon, number_of_points: int, rfactor : float):
    space = outline.bounds
    centerx, centery = outline.centroid.x, outline.centroid.y
    minr = poly_radius(bb_poly(outline.bounds))
    if number_of_points < 3:
        number_of_points=3
    if rfactor <= 1:
        rfactor = 1.5
    r = minr * rfactor
    arcstep = (2*pi)/number_of_points
    boundary_points=[]
    for t in range(0,number_of_points):
        x = r * cos(arcstep*t) + centerx
        y = r * sin(arcstep*t) + centery
        boundary_points.append((x,y))
    boundary_points=np.array(boundary_points)
    return boundary_points

def _voronoi_polygons_clip_to_outline(
        points: np.ndarray, 
        outline: Polygon, 
        bb_points: int, 
        bb_growth: float) -> list[Polygon]:
    """
    Build a Voronoi diagram for `points` and clip every cell to `outline`.

    Parameters
    ----------
    points : (N, 2) array
        The coordinates of the generators.
    outline : shapely.geometry.Polygon
        The shape that bounds all Voronoi cells.

    Returns
    -------
    List[shapely.geometry.Polygon]
        One polygon per point, clipped to the outline.
    """

    # Add new points to the supplied points list - this is to lessen the impact of
    # degenerate edges - basically, we surround the object of interest with a set
    # of generated points, so that all the voronoi cells inside have proper edges

    bb_points = _boundary_points(outline, bb_points, bb_growth)
    
    vor = Voronoi(np.vstack([points,bb_points]))
    polys: List[Polygon] = []

    for idx, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if -1 in vertices or len(vertices) == 0:
            # Infinite region – skip it (it will be discarded when clipping)
            continue

        region_pts = [vor.vertices[i] for i in vertices]
        poly = Polygon(region_pts)

        # Clip to the outline.  The clip may produce an empty geometry.
        poly = poly.intersection(outline)
        if poly.is_empty:
            continue
        # The intersection might be a MultiPolygon (rare for Voronoi cells)
        if isinstance(poly, MultiPolygon):
            # Keep the largest part – it is usually the whole cell
            poly = max(poly.geoms, key=lambda p: p.area)
            #poly = poly.intersection()
        polys.append(poly)

    return polys

# Hungarian Algorithm
def _cost_matrix(
    src: np.ndarray,
    tgt: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Build an m×n matrix C where C[i, j] = distance(s_i, t_j).
    Supports:
        - 'euclidean'  (default)
        - 'sqeuclidean'
        - any callable that takes two 2‑D arrays and returns a float matrix
    """
    if isinstance(metric, str):
        if metric == "euclidean":
            diff = src[:, None, :] - tgt[None, :, :]          # shape (m,n,2)
            C = np.sqrt(np.sum(diff**2, axis=2))
        elif metric == "special":
            diff = src[:, None, :] - tgt[None, :, :]          # shape (m,n,2)
            C = np.sqrt(np.sum(diff**2, axis=2))
        elif metric == "sqeuclidean":
            diff = src[:, None, :] - tgt[None, :, :]
            C = np.sum(diff**2, axis=2)
        else:
            raise ValueError(f"Unsupported metric string: {metric!r}")
    else:
        # user supplied custom metric
        C = metric(src[:, None, :], tgt[None, :, :])  # expecting shape (m,n)
    return C


