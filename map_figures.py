import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gridCoords_fromCorners(corners, grid_dims):
    if type(corners) is not np.ndarray:
        corners = np.array(corners)
    
    top_interp = np.empty([grid_dims[1], 2])
    right_interp = np.empty([grid_dims[0], 2])
    bottom_interp = np.empty([grid_dims[1], 2])
    left_interp = np.empty([grid_dims[0], 2])
    
    top_interp[[0,grid_dims[1]-1],:] = corners[[0,1],:]
    right_interp[[0,grid_dims[0]-1],:] = corners[[1,2],:]
    bottom_interp[[0,grid_dims[1]-1],:] = corners[[3,2],:]
    left_interp[[0,grid_dims[0]-1],:] = corners[[0,3],:]
    
    top_interp[1:(grid_dims[1]-1),0] = np.interp(np.arange(1, grid_dims[1]-1), [0, grid_dims[1]-1], corners[[0,1],0])
    top_interp[1:(grid_dims[1]-1),1] = np.interp(np.arange(1, grid_dims[1]-1), [0, grid_dims[1]-1], corners[[0,1],1])
    right_interp[1:(grid_dims[0]-1),0] = np.interp(np.arange(1, grid_dims[0]-1), [0, grid_dims[0]-1], corners[[1,2],0])
    right_interp[1:(grid_dims[0]-1),1] = np.interp(np.arange(1, grid_dims[0]-1), [0, grid_dims[0]-1], corners[[1,2],1])
    bottom_interp[1:(grid_dims[1]-1),0] = np.interp(np.arange(1, grid_dims[1]-1), [0, grid_dims[1]-1], corners[[3,2],0])
    bottom_interp[1:(grid_dims[1]-1),1] = np.interp(np.arange(1, grid_dims[1]-1), [0, grid_dims[1]-1], corners[[3,2],1])
    left_interp[1:(grid_dims[0]-1),0] = np.interp(np.arange(1, grid_dims[0]-1), [0, grid_dims[0]-1], corners[[0,3],0])
    left_interp[1:(grid_dims[0]-1),1] = np.interp(np.arange(1, grid_dims[0]-1), [0, grid_dims[0]-1], corners[[0,3],1])
    
    L2R_interp = np.empty(np.append(grid_dims, 2))
    L2R_interp[:,0,:] = left_interp
    L2R_interp[:,-1,:] = right_interp
    for i in range(grid_dims[0]):
        L2R_interp[i,np.arange(1, grid_dims[1]-1),0] = np.interp(np.arange(1, grid_dims[1]-1),             [0, grid_dims[1]-1], [left_interp[i,0], right_interp[i,0]])
        L2R_interp[i,np.arange(1, grid_dims[1]-1),1] = np.interp(np.arange(1, grid_dims[1]-1),             [0, grid_dims[1]-1], [left_interp[i,1], right_interp[i,1]])
    
    T2B_interp = np.empty(np.append(grid_dims, 2))
    T2B_interp[0,:,:] = top_interp
    T2B_interp[-1,:,:] = bottom_interp
    for i in range(grid_dims[1]):
        T2B_interp[np.arange(1, grid_dims[0]-1),i,0] = np.interp(np.arange(1, grid_dims[0]-1),             [0, grid_dims[0]-1], [top_interp[i,0], bottom_interp[i,0]])
        T2B_interp[np.arange(1, grid_dims[0]-1),i,1] = np.interp(np.arange(1, grid_dims[0]-1),             [0, grid_dims[0]-1], [top_interp[i,1], bottom_interp[i,1]])
    
    return (L2R_interp + T2B_interp)/2


def nearest_point_in_grid(target_coord, grid_coords):
    if type(target_coord) is not np.ndarray:
        target_coord = np.array(target_coord)
    if len(target_coord.shape) < 2:
        target_coord.reshape([-1,2])
    
    if len(grid_coords.shape) > 2:
        grid_flat = np.ravel(grid_coords).reshape([-1,2])
    else:
        grid_flat = grid_coords
    
    if len(target_coord.shape) < 2:
        distances = np.sqrt((target_coord[0] - grid_flat[:,0])**2 + (target_coord[1] - grid_flat[:,1])**2)
        target_inds = np.argmin(distances)
    else:
        target_inds = np.empty(target_coord.shape[0], dtype=int)
        for i in range(target_coord.shape[0]):
            distances = np.sqrt((target_coord[i,0] - grid_flat[:,0])**2 + (target_coord[i,1] - grid_flat[:,1])**2)
            target_inds[i] = np.argmin(distances)
    
    if len(grid_coords.shape) > 2:
        target_inds = np.unravel_index(target_inds, grid_coords.shape[0:2])
        if len(target_coord.shape) == 2:
            target_inds = zip(*target_inds)
    
    return target_inds


def arrayInds_from_LatLong(LatLong, LatLongGrid, grid_dims=None):
    if grid_dims is not None:
        LatLongGrid = gridCoords_fromCorners(LatLongGrid, grid_dims)
    
    arrayInds = nearest_point_in_grid(LatLong, LatLongGrid)
    
    return arrayInds


def colorMappable_from_values(values, lims=None, cmap=mpl.cm.jet):
    if not hasattr(values, 'size'):
        values = np.array(values)
    
    if lims is None:
        lims = [values.min(), values.max()]
    elif not hasattr(lims, '__len__'):
        print 'WARNING: lims argument is a scalar; ignoring and using max/min of values argument'
        lims = [values.min(), values.max()]
    elif len(lims) != 2:
        print 'WARNING: lims argument has >2 elements; ignoring and using max/min of values argument'
        lims = [values.min(), values.max()]
    cMappable = mpl.cm.ScalarMappable(mpl.colors.Normalize(lims[0], lims[1], clip=True), cmap)
    cMappable.set_array(values)
    return cMappable


class MapFig:
    def __init__(self, fig=None, ax=None, img=None, pts=None, cbar=None):
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()


def map_plot(map_img, points, LatLong=True, corners_LatLong=None, grid_LatLong=None, \
             values=None, colorMappable=None, alpha=None, marker='o', markersize=10, CBAR=True):
    if type(map_img) is str:
        map_img = plt.imread(fname=map_img)
    
    if map_img.shape[2] == 4:
        map_img = map_img[:,:,0:3]
    
    if type(corners_LatLong) is str:
        corners_LatLong = np.loadtxt(corners_LatLong, delimiter=',')
    
    if LatLong:
        if grid_LatLong is None:
            if corners_LatLong is None:
                raise RuntimeError('map_plot function must receive either corners_LatLong or grid_LatLong arument')
            else:
                points_inds = np.matrix(arrayInds_from_LatLong(points, corners_LatLong, map_img.shape[0:2]))
        else:
            points_inds = np.matrix(arrayInds_from_LatLong(points, grid_LatLong))
        
    elif type(points) is not np.matrix:
        points_inds = np.matrix(points)
    else:
        points_inds = points
    
    if type(values) is str:
        values = mpl.colors.colorConverter.to_rgba(values)
    
    if values is None:
        values = [mpl.colors.colorConverter.to_rgba('b') for i in range(points_inds.shape[0])]
    elif type(values) is tuple:
        if len(values) == 3:
            values = tuple(list(values) + [1])
        values = [values for i in range(points_inds.shape[0])]
    elif type(values) is not np.ndarray:
        values = np.array(values)
    
    if type(values) is np.ndarray:
        if colorMappable is None:
            colorMappable_from_values(values)
        colors = colorMappable.to_rgba(values)
        colors = [tuple(colors[i,:]) for i in range(colors.shape[0])]
    else:
        colors = values
        colorMappable = None
    
    # Transparency
    if alpha is not None:
        for i in range(len(colors)):
            colors[i] = tuple(list(colors[i][:3]) + [alpha])
    
    mfig = MapFig()
    mfig.img = plt.imshow(map_img)
    mfig.fig = plt.gcf()
    mfig.ax = plt.gca()
    mfig.ax.set_frame_on(False)
    mfig.ax.axes.get_xaxis().set_visible(False)
    mfig.ax.axes.get_yaxis().set_visible(False)
    
    mfig.pts = []
    for i in range(points_inds.shape[0]):
        mfig.pts.append(plt.plot(points_inds[i,1], points_inds[i,0], color=colors[i], \
            marker=marker, markersize=markersize, markeredgecolor=None))
    
    if CBAR and (colorMappable is not None):
        divider = make_axes_locatable(mfig.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        mfig.cbar = plt.colorbar(colorMappable, cax)
    
    mfig.ax.margins(0)
    mfig.fig.tight_layout()
    return(mfig)


def create_circles(xy_tuples, radii=None, colors_or_cmapNormInd=None):
    if radii is None:
        radii = np.zeros(len(xy_tuples))
    else:
        radii = np.array(radii)
    
    if radii.size == 1:
        radii = np.ones(len(xy_tuples)) * radii
    else:
        radii = np.array(radii)
    
    # Convert color argument to list of matplotlib color specs
    if (colors_or_cmapNormInd is None) or (len(colors_or_cmapNormInd) == 0):
        colors = ['b' for i in range(len(xy_tuples))]
    elif len(colors_or_cmapNormInd) == 1:
        colors = [colors_or_cmapNormInd for i in range(len(xy_tuples))]
    else:
        colors = np.array(colors_or_cmapNormInd)
        colors = mpl.cm.jet(np.int64(255*colors))
    
    circles = []
    for i in range(len(xy_tuples)):
        circles.append( plt.Circle(xy_tuples[i], radii[i], color=colors[i]) )
    
    return circles


def circle_xylims(circles):
    xvals = np.concatenate([circles[i].center[0] + circles[i].radius*np.array([-1, 1]) for i in range(len(circles))])
    xlims = [xvals.min(), xvals.max()]
    yvals = np.concatenate([circles[i].center[1] + circles[i].radius*np.array([-1, 1]) for i in range(len(circles))])
    ylims = [yvals.min(), yvals.max()]
    return np.concatenate([xlims, ylims])


def draw_circles(circles, figORax):
    if 'matplotlib.figure.Figure' in str(type(figORax)):
        ax = figORax.gca()
    elif 'matplotlib.axes.Axes' in str(type(figORax)):
        ax = figORax
    else:
        fig = plt.figure()
        ax = fig.gca()
    
    for i in range(len(circles)):
        ax.add_artist(copy.copy(circles[i]))
    
    return ax
