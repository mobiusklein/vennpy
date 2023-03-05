from typing import Any, Dict, Set, List, Optional, Union, Tuple
from string import ascii_uppercase
from itertools import cycle

from dataclasses import dataclass, field

from matplotlib import pyplot as plt, colors as mcolor
from matplotlib.axes import Axes
from matplotlib.patches import Circle as MCircle

from .set_logic import BaseSet, VSet, combinate_sets, SetCombination, T
from .layout import GreedyLayout, RefinedLayout, Overlap
from .geometry import Point, Circle


ColorLike = Union[str, Tuple[float, float, float]]


COLORS: List[ColorLike] = list(map(mcolor.hex2color, [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]))

ASCII_UPPERCASE = list(ascii_uppercase)


class VennDiagram:
    sets: List[Set[T]]
    names: List[str]

    layout: RefinedLayout

    ax: Axes

    colors: List[ColorLike]
    linewight: float = 2
    alpha: float = 0.5
    fill: bool = True
    circle_kwargs: Dict[str, Any]

    include_label: bool = True
    include_size: bool = True
    label_size: float = 12



def venn_layout(sets: List[Set[T]], names: Optional[List[str]]=None, normalize: bool=False):

    if names is None:
        names = ASCII_UPPERCASE[:len(sets)]
        if len(names) < len(sets):
            raise ValueError("Too many sets, please provide explicit names")
    vsets = []
    for s, name in zip(sets, names):
        vsets.append(VSet(name, s))

    combos = combinate_sets(vsets)
    initial = GreedyLayout(combos)
    refined = RefinedLayout.from_greedy(initial, normalize=normalize)
    return refined


def draw_circles(ax: Axes, refined: RefinedLayout, colors: List[ColorLike],
                 lineweight: int=2, alpha: float=0.5, fill: bool=True, **kwargs):

    patches = []
    for i, (_k, circle) in enumerate(refined.circles.items()):
        patch = MCircle((circle.x, circle.y), radius=circle.radius,
                        facecolor=colors[i] if fill else 'none',
                        edgecolor=colors[i], lw=lineweight,
                        label=circle.label,
                        alpha=alpha, **kwargs)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def draw_labels(ax: Axes, refined: RefinedLayout, include_label: bool = True,
                include_size: bool = True):
    placed: List[Tuple[BaseSet, Point]] = []
    artists = []
    for s in sorted(refined.sets, key=lambda x: refined.exclusive_sizes[x.name], reverse=True):
        pt = refined.centers[s.name]
        size_of_set = refined.exclusive_sizes[s.name]
        skip = False
        if size_of_set == 0:
            for (prev_s, prev_pt) in placed:
                if pt.distance(prev_pt) < 0.1:
                    skip = True
                    break
                if len(prev_s.component_overlaps(s)) >= s.degree:
                    skip = True
                    break

        if skip:
            continue

        label = None
        if include_label and include_size:
            label = f"{s.name} {size_of_set}"
        elif include_label:
            label = s.name
        elif include_size:
            label = str(size_of_set)

        if label:
            artists.append(ax.text(pt.x, pt.y, label, ha='center'))
        placed.append((s, pt))
    return artists


def venn(sets: List[Set[T]], names: Optional[List[str]]=None, colors: Optional[List[ColorLike]]=None,
         ax: Optional[Axes]=None, alpha: float=0.5, fill=True, lineweight: int=2, include_label: bool=True,
         include_size: bool=True, __normalize: bool=False):

    if colors is None:
        gen = cycle(COLORS)
        colors = [next(gen) for _ in sets]

    refined = venn_layout(sets, names, __normalize)

    if ax is None:
        _fig, ax = plt.subplots(1, 1)

    patches = draw_circles(ax, refined, colors, lineweight=lineweight, alpha=alpha, fill=fill)

    text_labels = draw_labels(ax, refined, include_label, include_size)
    ax.autoscale()
    ax.axis('off')
    return ax, [patches, text_labels], refined

