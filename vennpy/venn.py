from typing import Set, List, Optional, Union, Tuple
from string import ascii_uppercase
from itertools import cycle

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


def venn_layout(sets: List[Set[T]], names: Optional[List[str]]=None):

    if names is None:
        names = ASCII_UPPERCASE[:len(sets)]
        if len(names) < len(sets):
            raise ValueError("Too many sets, please provide explicit names")
    vsets = []
    for s, name in zip(sets, names):
        vsets.append(VSet(name, s))

    combos = combinate_sets(vsets)
    initial = GreedyLayout(combos)
    refined = RefinedLayout(initial.sets, initial.circles)
    return refined



def venn(sets: List[Set[T]], names: Optional[List[str]]=None, colors: Optional[List[ColorLike]]=None,
         ax: Optional[Axes]=None, alpha: float=0.5, fill=True, include_label: bool=True,
         include_size: bool=True):

    if colors is None:
        gen = cycle(COLORS)
        colors = [next(gen) for _ in sets]

    refined = venn_layout(sets, names)

    if ax is None:
        _fig, ax = plt.subplots(1, 1)

    patches = []
    for i, (_k, circle) in enumerate(refined.circles.items()):
        patch = MCircle((circle.x, circle.y), radius=circle.radius,
                        facecolor=colors[i] if fill else 'none',
                        edgecolor=colors[i], lw=2,
                        label=circle.label,
                        alpha=alpha)
        ax.add_patch(patch)
        patches.append(patch)

    placed: List[Point] = []
    for s in sorted(refined.sets, key=lambda x: refined.exclusive_sizes[x.name], reverse=True):
        pt = refined.centers[s.name]

        skip = False
        for p in placed:
            if pt.distance(p) < 0.1 and refined.exclusive_sizes[s.name] == 0:
                skip = True
                break
        if skip:
            continue


        label = None
        if include_label and include_size:
            label = f"{s.name} {refined.exclusive_sizes[s.name]}"
        elif include_label:
            label = s.name
        elif include_size:
            label = str(refined.exclusive_sizes[s.name])

        if label:
            ax.text(pt.x, pt.y, label, ha='center')
        placed.append(pt)
    ax.autoscale()
    return ax, patches, refined

