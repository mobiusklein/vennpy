from cProfile import label
from typing import Hashable, List, Set, Optional
from string import ascii_uppercase
from itertools import cycle, product

from .circle import Point, Circle, Arc
from .layout import GreedyLayout, Group, Overlap, compute_text_centers

from matplotlib import pyplot as plt
from matplotlib import colors as mcolor

COLORS = list(map(mcolor.hex2color, [
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


def venn(subsets: List[Set[Hashable]], set_labels: Optional[List[str]]=None, colors: Optional[List[str]]=None,
         ax: Optional[plt.Axes]=None, scale: float=1.0):
    standalone = False
    if ax is None:
        standalone = True
        _fig, ax = plt.subplots(1, 1)
    if set_labels is None:
        set_labels = ASCII_UPPERCASE[:len(subsets)]

    if colors is None:
        gen = cycle(COLORS)
        colors = [next(gen) for i in range(len(subsets))]

    groups = []
    color_map = {}
    for group, c in [(Group(s, lab), c) for s, lab, c in zip(subsets, set_labels, colors)]:
        groups.append(group)
        color_map[group.label] = c

    layout = GreedyLayout(groups, scale)

    text_centers = compute_text_centers(layout)
    for k, v in layout.circles.items():
        c = plt.Circle((v.x, v.y), v.radius, alpha=0.5, label=k, facecolor=color_map[k])
        ax.add_patch(c)

    for k, v in text_centers.items():
        k = '|'.join(k)
        ax.text(v.x, v.y, k, ha='center')


    bbox = layout.bounding_box()
    ax.set_xlim(*bbox['x'])
    ax.set_ylim(*bbox['y'])
    ax.axis("off")
    return ax
