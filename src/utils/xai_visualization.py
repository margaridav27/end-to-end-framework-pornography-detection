from src.utils.misc import unnormalize

import warnings
from typing import List, Tuple, Optional, Any, Union
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import axis, figure
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch


VISUALIZATION_TYPES = {
    "original_image",
    "heat_map",
    "blended_heat_map",
    "masked_image",
    "alpha_scaling",
}

SIGN_TYPES = {"all", "positive", "negative", "absolute_value"}


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def _prepare_image(attr_visual: np.ndarray):
    return np.clip(attr_visual.astype(int), 0, 255)


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def _normalize_scale(attr: np.ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by a scale factor of 0"

    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results may be misleading. "
            "This likely means that attribution values are all close to 0."
        )

    return np.clip(attr / scale_factor, -1, 1)


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def _cumulative_sum_threshold(values: np.ndarray, percentile: Union[int, float]):
    assert percentile >= 0 and percentile <= 100, "Percentile for thresholding must be " "between 0 and 100 inclusive."

    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]

    return sorted_vals[threshold_id]


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def normalize_attr(
    attr: np.ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    # attr_combined = attr
    if reduction_axis is not None:
        attr = np.sum(attr, axis=reduction_axis)

    percentile = 100 - outlier_perc

    # Choose appropriate signed values and rescale, removing given outlier percentage
    if sign == "all":
        threshold = _cumulative_sum_threshold(np.abs(attr), percentile)
    elif sign == "positive":
        attr = np.maximum(attr, 0)  # Keep only positive values
        threshold = _cumulative_sum_threshold(attr, percentile)
    elif sign == "negative":
        attr = np.minimum(attr, 0)  # Keep only negative values
        threshold = -1 * _cumulative_sum_threshold(np.abs(attr), percentile)
    elif sign == "absolute_value":
        attr = np.abs(attr)
        threshold = _cumulative_sum_threshold(attr, percentile)

    return _normalize_scale(attr, threshold)


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def visualize_image_attr(
    attr: np.ndarray,
    original_image: Union[None, np.ndarray] = None,
    method: str = "heat_map",
    sign: str = "absolute_value",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):
    r"""
    Visualizes attribution for a given image by normalizing attribution values
    of the desired sign (positive, negative, absolute value, or all) and displaying
    them using the desired mode in a matplotlib figure.

    Args:

        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (H, W, C), with
                    channels as last dimension. Shape must also match that of
                    the original image if provided.
        original_image (numpy.ndarray, optional): Numpy array corresponding to
                    original image. Shape must be in the form (H, W, C), with
                    channels as the last dimension. Image can be provided either
                    with float values in range 0-1 or int values between 0-255.
                    This is a necessary argument for any visualization method
                    which utilizes the original image.
                    Default: None
        method (str, optional): Chosen method for visualizing attribution.
                    Supported options are:

                    1. `heat_map` - Display heat map of chosen attributions

                    2. `blended_heat_map` - Overlay heat map over greyscale
                       version of original image. Parameter alpha_overlay
                       corresponds to alpha of heat map.

                    3. `original_image` - Only display original image.

                    4. `masked_image` - Mask image (pixel-wise multiply)
                       by normalized attribution values.

                    5. `alpha_scaling` - Sets alpha channel of each pixel
                       to be equal to normalized attribution value.

                    Default: `heat_map`
        sign (str, optional): Chosen sign of attributions to visualize. Supported
                    options are:

                    1. `positive` - Displays only positive pixel attributions.

                    2. `absolute_value` - Displays absolute value of
                       attributions.

                    3. `negative` - Displays only negative pixel attributions.

                    4. `all` - Displays both positive and negative attribution
                       values. This is not supported for `masked_image` or
                       `alpha_scaling` modes, since signed information cannot
                       be represented in these modes.

                    Default: `absolute_value`
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
                    Default: None
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        cmap (str, optional): String corresponding to desired colormap for
                    heatmap visualization. This defaults to "Reds" for negative
                    sign, "Blues" for absolute value, "Greens" for positive sign,
                    and a spectrum from red to green for all. Note that this
                    argument is only used for visualizations displaying heatmaps.
                    Default: None
        alpha_overlay (float, optional): Alpha to set for heatmap when using
                    `blended_heat_map` visualization mode, which overlays the
                    heat map over the greyscaled original image.
                    Default: 0.5
        show_colorbar (bool, optional): Displays colorbar for heatmap below
                    the visualization. If given method does not use a heatmap,
                    then a colormap axis is created and hidden. This is
                    necessary for appropriate alignment when visualizing
                    multiple plots, some with colorbars and some without.
                    Default: False
        title (str, optional): Title string for plot. If None, no title is
                    set.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (6,6)
        use_pyplot (bool, optional): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.

    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> ig = IntegratedGradients(net)
        >>> # Computes integrated gradients for class 3 for a given image .
        >>> attribution, delta = ig.attribute(orig_image, target=3)
        >>> # Displays blended heat map visualization of computed attributions.
        >>> _ = visualize_image_attr(attribution, orig_image, "blended_heat_map")
    """

    if method not in VISUALIZATION_TYPES:
        raise ValueError(f"Invalid visualization method '{method}'.")
    if sign not in SIGN_TYPES:
        raise ValueError(f"Invalid visualization sign '{sign}'.")

    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    # Original image must be provided for any visualization other than heatmap
    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = _prepare_image(original_image * 255)
    elif method != "heat_map":
        raise ValueError("Original image must be provided for any visualization other than heatmap.")

    # Remove ticks and tick labels from plot
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(visible=False)

    heat_map = None

    # Show original image
    if method == "original_image":
        if len(original_image.shape) > 2 and original_image.shape[2] == 1:
            original_image = np.squeeze(original_image, axis=2)
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize
        norm_attr = normalize_attr(attr, sign, outlier_perc, reduction_axis=2)

        # Set default colormap and bounds based on sign
        if sign == "all":
            default_cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
            vmin, vmax = -1, 1
        elif sign == "positive":
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif sign == "negative":
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif sign == "absolute_value":
            default_cmap = "Blues"
            vmin, vmax = 0, 1

        cmap = cmap if cmap is not None else default_cmap

        if method == "heat_map":
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif method == "blended_heat_map":
            plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay)
        elif method == "masked_image":
            assert (
                sign != "all"
            ), "Cannot display masked image with both positive and negative attributions, choose a different sign."
            plt_axis.imshow(_prepare_image(original_image * np.expand_dims(norm_attr, axis=2)))
        elif method == "alpha_scaling":
            assert (
                sign != "all"
            ), "Cannot display alpha scaling with both positive and negative attributions, choose a different sign."
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        _prepare_image(np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )

    # If given method is not a heatmap and no colormap is relevant, then a colormap axis is created and hidden
    # This is necessary for appropriate alignment when visualizing multiple plots, some with heatmaps and some without
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")

    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


# Source: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
def visualize_image_attr_multiple(
    attr: np.ndarray,
    original_image: Union[None, np.ndarray],
    methods: List[str],
    signs: List[str],
    titles: Union[None, List[str]] = None,
    fig_size: Tuple[int, int] = (8, 6),
    use_pyplot: bool = True,
    **kwargs: Any,
):
    r"""
    Visualizes attribution using multiple visualization methods displayed
    in a 1 x k grid, where k is the number of desired visualizations.

    Args:

        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (H, W, C), with
                    channels as last dimension. Shape must also match that of
                    the original image if provided.
        original_image (numpy.ndarray, optional): Numpy array corresponding to
                    original image. Shape must be in the form (H, W, C), with
                    channels as the last dimension. Image can be provided either
                    with values in range 0-1 or 0-255. This is a necessary
                    argument for any visualization method which utilizes
                    the original image.
        methods (list[str]): List of strings of length k, defining method
                        for each visualization. Each method must be a valid
                        string argument for method to visualize_image_attr.
        signs (list[str]): List of strings of length k, defining signs for
                        each visualization. Each sign must be a valid
                        string argument for sign to visualize_image_attr.
        titles (list[str], optional): List of strings of length k, providing
                    a title string for each plot. If None is provided, no titles
                    are added to subplots.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (8, 6)
        use_pyplot (bool, optional): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        **kwargs (Any, optional): Any additional arguments which will be passed
                    to every individual visualization. Such arguments include
                    `show_colorbar`, `alpha_overlay`, `cmap`, etc.


    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> ig = IntegratedGradients(net)
        >>> # Computes integrated gradients for class 3 for a given image .
        >>> attribution, delta = ig.attribute(orig_image, target=3)
        >>> # Displays original image and heat map visualization of
        >>> # computed attributions side by side.
        >>> _ = visualize_image_attr_multiple(attribution, orig_image,
        >>>                     ["original_image", "heat_map"], ["all", "positive"])
    """
    
    assert len(methods) == len(signs), "Methods and signs array lengths must match."

    if titles is not None:
        assert len(methods) == len(titles), "If titles list is given, length must match that of methods list."
    
    if use_pyplot:
        plt_fig = plt.figure(figsize=fig_size)
    else:
        plt_fig = Figure(figsize=fig_size)
    
    plt_axis = plt_fig.subplots(1, len(methods))

    # When visualizing one
    if len(methods) == 1:
        plt_axis = [plt_axis]

    for i in range(len(methods)):
        visualize_image_attr(
            attr,
            original_image=original_image,
            method=methods[i],
            sign=signs[i],
            plt_fig_axis=(plt_fig, plt_axis[i]),
            use_pyplot=False,
            title=titles[i] if titles else None,
            **kwargs,
        )
    
    plt_fig.tight_layout()
    
    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


def visualize_explanation(
    image: Union[torch.Tensor, np.ndarray],
    attr: Union[np.ndarray, str],
    sign: str = "positive",
    method: str = "blended_heat_map",
    colormap: Optional[str] = None,
    outlier_perc: Union[float, int] = 2,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    side_by_side: bool = False,
    **kwargs,
):
    CHW = lambda shape: shape[0] in (3, 4)
    HWC = lambda input: np.transpose(input, (1, 2, 0))

    # Convert attr to numpy array of shape (H, W, C)
    if isinstance(attr, str): attr = np.load(attr)
    if len(attr.shape) == 4: attr = np.squeeze(attr)
    if len(attr.shape) == 2: attr = np.expand_dims(attr, axis=-1)
    if CHW(attr.shape): attr = HWC(attr)

    # Convert image to numpy array of shape (H, W, C)
    if torch.is_tensor(image): image = image.cpu().numpy()
    if len(image.shape) == 4: image = np.squeeze(image)
    if CHW(image.shape): image = HWC(image)

    # assert image.shape[:2] == attr.shape[:2], "Image and attr shapes must match."

    if side_by_side:
        norm_std = kwargs.get("norm_std", None)
        norm_mean = kwargs.get("norm_mean", None)

        # Unnormalize for better visualization
        if norm_std and norm_mean:
            image = unnormalize(image, norm_mean, norm_std)

        return visualize_image_attr_multiple(
            attr=attr,
            original_image=image,
            methods=["original_image", method],
            signs=["all", sign],
            cmap=colormap,
            show_colorbar=show_colorbar,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
        )[0]
    else:
        return visualize_image_attr(
            attr=attr,
            original_image=image,
            method=method,
            sign=sign,
            cmap=colormap,
            show_colorbar=show_colorbar,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
        )[0]
