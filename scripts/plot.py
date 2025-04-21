import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlate2d
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from math import factorial

__all__ = ['turbo_w', 'make_theme', 'plot_image', 'enlarge', 'sgolay', 'sgolay2d']

def turbo_w():
    """
    modified turbo colormap with white background
    cmap = turbo_w()
    """
    cmap = np.load('cmaps/white_turbo.npy')
    return ListedColormap(cmap)

def make_theme(ax, logx=False, logy=False, grid=False, minorx=2, minory=2, ticklength=6, lw=0.5):
    """ Apply the theme for figures
    
    Parameters
    ----------
    ax :  AxesSubplot
        AxesSubplot axes.
    logx :  bool
        Where use log scale for x axis (default: False).
    logy :  bool
        Where use log scale for y axis (default: False).
    grid :  bool, string, list or dict
        Grid parameters. If False, grid will not show (True not supported).
        If string, it will be used as line style for grid of major ticks.
        If list, it will be used as line styles for grid of major and minor ticks, respectively.
        If dict, it can contain some of the keys including 'lw', 'ls' and 'color',
            each of which can be either a string or a list.
            If one of them is given as a list, minor grid lines will be shown (default: False).
    minorx :  bool, int
        Whether to show minor x ticks. If False, it will not show (True not supported, default: 2)
    minory :  bool, int
        Whether to show minor y ticks. If False, it will not show (True not supported, default: 2)
    ticklength :  int
        Length of axis ticks (default: 6).
    lw :  float
        Line width for both the ticks and frame (default: 0.5).
    """
    if logx:
        ax.set_xscale('log')
        minorx = False
    if minorx:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(int(minorx)))
    if logy:
        ax.set_yscale('log')
        minory = False
    if minory:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(int(minory)))
    if not logx and not logy:
        ax.ticklabel_format(useOffset=False, style='plain')
    if grid:
        ngrid = 1
        if isinstance(grid, dict):
            if 'ls' not in grid.keys():
                gls = ['--', '--']
            elif isinstance(grid['ls'], list):
                gls = grid['ls']
                ngrid = 2
            else:
                gls = [grid['ls'], grid['ls']]
            if 'lw' not in grid.keys():
                glw = [0.5, 0.5]
            elif isinstance(grid['lw'], list):
                glw = grid['lw']
                ngrid = 2
            else:
                glw = [grid['lw'], grid['lw']]
            if 'color' not in grid.keys():
                gcolor = ['gray', 'lightgray']
            elif isinstance(grid['color'], list):
                gcolor = grid['color']
                ngrid = 2
            else:
                gcolor = [grid['color'], grid['color']]
        elif isinstance(grid, list):
            gls = grid
            ngrid = 2
            glw = [0.5, 0.5]
            gcolor = ['gray', 'lightgray']
        else:
            gls = [grid, grid]
            glw = [0.5, 0.5]
            gcolor = ['gray', 'lightgray']
        ax.grid(ls=gls[0], color=gcolor[0], lw=glw[0], which='major')
        if ngrid==2:
            ax.grid(ls=gls[1], color=gcolor[1], lw=glw[1], which='minor')
    
    ax.tick_params(which='major', direction='in', left=True, bottom=True, top=True, right=True, width=lw, length=ticklength)
    ax.tick_params(which='minor', direction='in', left=minory, bottom=minorx, top=minorx, right=minory, width=lw, length=ticklength/2)
    ax.ticklabel_format(useOffset=False, style='plain')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)

def plot_image(image, x=None, y=None, ax=None, cax=None, imshow=False,
               vmin=None, vmax=None, pmin=1, pmax=99, aspect='auto',
               cbar=False, cbar_label=None, cmap='viridis'):
    """ Plot 2D Image.
    
    Parameters
    ----------
    image :  2d array, image to plot.
    x :  1d array, list of x axis coordinates. If None, imshow is called (default: None).
    y :  1d array, list of y axis coordinates. If None, imshow is called (default: None).
    ax :  AxesSubplot, ax to plot main figure. If None, a new ax will be made (default: None).
    cax :  AxesSubplot, ax to plot colorbar. If None, colorbar will be plot in ax (default: None).
    imshow :  bool, enforce imshow function even if x or y is given (default: False).
    vmin, vmax :  float, data range that the colormap covers. If None, they will be determined by pmin and pmax (default: None).
    pmin, pmax :  float, data percentage rage that the colormap covers (default: 1).
    aspect :  str or array, the aspect ratio of the Axes (default: auto).
    cbar :  bool, plot the colorbar (default: False).
    cbar_label :  str, label of colorbar (default: None).
    cmap :  str, colormap (default: viridis).
    
    Returns
    -------
    ax, art, cart if cbar is True
    ax, art if cbar is False
    """
    if ax is None:
        fig, ax = plt.subplots()
    if vmin == None:
        vmin = np.nanpercentile(image, pmin)
    if vmax == None:
        vmax = np.nanpercentile(image, pmax)
    if x is None or y is None:
        imshow = True
    if imshow:
        if x is None and y is None:
            art = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation=None, aspect=aspect, origin=None, cmap=cmap, rasterized=True)
        elif x is None:
            art = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation=None, aspect=aspect, origin='lower', cmap=cmap, rasterized=True,
                            extent=[0, np.array(image).shape[1]-1, np.min(y), np.max(y)])
        elif y is None:
            art = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation=None, aspect=aspect, origin='lower', cmap=cmap, rasterized=True,
                            extent=[np.min(x), np.max(x), 0, np.array(image).shape[0]-1])
        else:
            art = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation=None, aspect=aspect, origin='lower', cmap=cmap, rasterized=True,
                            extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    else:
        if x is None:
            x = np.arange(np.array(image).shape[1])
        if y is None:
            y = np.arange(np.array(image).shape[0])
        art = ax.pcolormesh(enlarge(x), enlarge(y), image, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        ax.axis(aspect)
    if cbar:
        if cax is None:
            cart = plt.colorbar(art, ax=ax)
        else:
            cart = plt.colorbar(art, cax=cax)
        if cbar_label is not None:
            cart.set_label(cbar_label)
        cart.outline.set_linewidth(0.5)
        return ax, art, cart
    else:
        return ax, art

def enlarge(x0):
    """ Extend the axis: enlarge(vector) """
    if isinstance(x0, list):
        x = np.array(x0)
    else:
        x = x0-0
    new_x = np.zeros(np.size(x)+1)
    new_x[1:-1] = (x[1:]+x[:-1])/2
    new_x[0] = 3./2.*x[0]-1./2.*x[1]
    new_x[-1] = 3./2.*x[-1]-1./2.*x[-2]
    return new_x


def sgolay(y, window_size, order, deriv=0, rate=1):
    r"""
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def sgolay2d(image, window_size, order, derivative=None):
    """
    Savitzky-Golay filter for 2D image arrays.
    See: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    Parameters
    ----------
    image : ndarray, shape (N,M)
        image to be smoothed.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    smooth_image : ndarray, shape (N,M)
        the smoothed image .
    """
    
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = image.shape[0] + 2*half_size, image.shape[1] + 2*half_size
    smooth_image = np.zeros( (new_shape) )
    # top band
    band = image[0, :]
    smooth_image[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( image[1:half_size+1, :] ) - band )
    # bottom band
    band = image[-1, :]
    smooth_image[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( image[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( image[:,0].reshape(-1,1), [1,half_size])
    smooth_image[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( image[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( image[:,-1].reshape(-1,1), [1,half_size] )
    smooth_image[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( image[:, -half_size-1:-1] ) - band )
    # central band
    smooth_image[half_size:-half_size, half_size:-half_size] = image

    # top left corner
    band = image[0,0]
    smooth_image[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(image[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = image[-1,-1]
    smooth_image[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(image[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = smooth_image[half_size,-half_size:]
    smooth_image[:half_size,-half_size:] = band - np.abs( np.flipud(smooth_image[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = smooth_image[-half_size:,half_size].reshape(-1,1)
    smooth_image[-half_size:,:half_size] = band - np.abs( np.fliplr(smooth_image[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(smooth_image, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(smooth_image, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(smooth_image, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(smooth_image, -r, mode='valid'), fftconvolve(smooth_image, -c, mode='valid')
