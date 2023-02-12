import math
import warnings

from scipy.optimize import bisect, minimize

from typing import Any, Iterable, Optional, List, Set, Tuple, DefaultDict, Dict, FrozenSet, Union
from dataclasses import dataclass, field

import numpy as np

from .geometry import Point, Circle, Arc, SMALL, circle_overlap
from .set_logic import BaseSet, VSet, SetCombination


@dataclass(frozen=True)
class Overlap:
    sets: FrozenSet[str]
    size: int
    weight: float = field(default=0.0)

    @classmethod
    def from_combo(cls, combo: SetCombination, weight: float=1):
        return cls(frozenset([v.key for v in combo.sets]), combo.cardinality, weight)

    @classmethod
    def from_vset(cls, vset: VSet, weight: float=1):
        return cls(frozenset((vset.key, )), vset.cardinality, weight)

    @classmethod
    def from_setlike(cls, setlike: BaseSet, weight: float=1):
        if setlike.degree == 1:
            return cls.from_vset(setlike, weight)
        else:
            return cls.from_combo(setlike, weight)

    @property
    def degree(self) -> int:
        return len(self.sets)


@dataclass(frozen=True)
class SetMeasure:
    key: str
    size: int
    weight: float = 1.0



@dataclass
class GreedyLayout:
    sets: List[BaseSet]
    circles: Dict[str, Circle] = field(default_factory=dict, init=False)
    set_overlaps: Dict[str, List[SetMeasure]] = field(
        default_factory=dict, init=False)
    pairwise_overlaps: List[Overlap] = field(default_factory=list, init=False)
    most_overlapped: List[SetMeasure] = field(default_factory=list, init=False)
    positioned: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.initialize_circles()
        self.build_overlaps()
        self.position_circles()

    def initialize_circles(self):
        for s in self.sets:
            if isinstance(s, VSet):
                self.circles[s.key] = Circle(
                    1e10,
                    1e10,
                    s.name,
                    size=s.cardinality,
                    radius=math.sqrt(s.cardinality / math.pi))
                self.set_overlaps[s.key] = []

    def build_overlaps(self):
        self.pairwise_overlaps = []
        for s in self.sets:
            if isinstance(s, SetCombination) and len(s.sets) == 2:

                weight = 1.0
                left, right = s.sets

                if s.cardinality + 1e-10 > min(left.cardinality, right.cardinality):
                    weight = 0.0

                self.pairwise_overlaps.append(Overlap.from_combo(s, weight))
                self.set_overlaps[left.key].append(
                    SetMeasure(right.key, s.cardinality, weight))
                self.set_overlaps[right.key].append(
                    SetMeasure(left.key, s.cardinality, weight))

        self.most_overlapped: List[SetMeasure] = []
        for k, overlaps in self.set_overlaps.items():
            size = sum(o.size * o.weight for o in overlaps)
            self.most_overlapped.append(SetMeasure(k, size))

        self.most_overlapped.sort(key=lambda x: x.size, reverse=True)

    def position_set(self, point: Point, key: str):
        self.circles[key].x = point.x
        self.circles[key].y = point.y
        self.positioned.add(key)

    def position_circles(self):
        self.position_set(Point(0, 0), self.most_overlapped[0].key)

        for s in self.most_overlapped[1:]:
            overlap = [o for o in self.set_overlaps[s.key] if o.key in self.positioned]
            overlap.sort(key=lambda x: x.size, reverse=True)
            circle_of_s = self.circles[s.key]
            if not overlap:
                raise ValueError("Missing pairwise overlap information!")

            points: List[Point] = []
            for j, o in enumerate(overlap):
                p1 = self.circles[o.key]

                # approximate distance from most overlapped already added set
                d1 = distance_from_intersection(
                    circle_of_s.radius, p1.radius, o.size)

                # sample positions at 90 degrees for looks
                points.append(Point(p1.x + d1, p1.y))
                points.append(Point(p1.x - d1, p1.y))
                points.append(Point(p1.x, p1.y + d1))
                points.append(Point(p1.x, p1.y - d1))

                # Get possible points relative to a third overlapping
                # circle
                for o2 in overlap[j + 1:]:
                    p2 = self.circles[o2.key]
                    d2 = distance_from_intersection(
                        circle_of_s.radius,
                        p2.radius,
                        o2.size
                    )

                    points.extend(
                        Circle(p1.x, p1.y, radius=d1).intersection_points(
                            Circle(p2.x, p2.y, radius=d2)
                        )
                    )

                best_loss = 1e50
                best_point = points[0]
                for point in points:
                    circle_of_s.x = point.x
                    circle_of_s.y = point.y
                    local_loss = loss_function(self.circles, self.pairwise_overlaps)
                    if local_loss < best_loss:
                        best_loss = local_loss
                        best_point = point
                self.position_set(best_point, s.key)


@dataclass
class RefinedLayout:
    sets: List[BaseSet]
    circles: Dict[str, Circle]
    overlaps: List[Overlap] = field(default_factory=list, init=False)
    multiple_overlaps: List[Overlap] = field(default_factory=list, init=False)
    set_ids: List[str] = field(default_factory=list, init=False)
    centers: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.overlaps = [Overlap.from_setlike(v) for v in self.sets]
        self.multiple_overlaps = list(filter(lambda x: x.degree > 1, self.overlaps))
        self.refine_set_positions()
        self.refine_text_centers()

    def _optimization_objective(self, values: List[float]):
        current = {}
        for i, set_id in enumerate(self.set_ids):
            current[set_id] = Circle(
                values[2 * i], values[2 * i + 1], set_id, self.circles[set_id].radius)
        return loss_function(current, self.multiple_overlaps)

    def refine_set_positions(self):
        self.set_ids = []
        initials = []
        for set_id, circle in self.circles.items():
            initials.append(circle.x)
            initials.append(circle.y)
            self.set_ids.append(set_id)

        solution = minimize(self._optimization_objective, initials, method='nelder-mead')
        selected_coordinates = solution.x

        for i, set_id in enumerate(self.set_ids):
            self.circles[set_id].x = selected_coordinates[2 * i]
            self.circles[set_id].y = selected_coordinates[2 * i + 1]

    def refine_text_centers(self):
        centers = compute_text_centers(self.circles, self.overlaps)
        for s in self.sets:
            pt = centers[Overlap.from_setlike(s).sets]
            self.centers[s.name] = pt


def distance_from_intersection(r1: float, r2: float, overlap: float) -> float:
    min_r = min(r1, r2)
    if min_r ** 2 * np.pi <= overlap + SMALL:
        return abs(r1 - r2)
    return bisect(lambda distance: circle_overlap(r1, r2, distance) - overlap, 0, r1 + r2)


def loss_function(circles: Dict[str, Circle], overlaps: List[Overlap]) -> float:
    output = 0.0
    for area in overlaps:
        overlap = 0.0
        if (len(area.sets) == 1 and isinstance(area.sets, frozenset)) or isinstance(area.sets, str):
            continue
        if len(area.sets) == 2:
            left_key, right_key = area.sets
            left: Circle = circles[left_key]
            right = circles[right_key]
            overlap = left.overlap(right)
        else:
            overlap = Circle.group_intersection_area(
                [circles[c] for c in area.sets])
        output += (overlap - area.size) ** 2
    return output


def find_overlapping_circles(circles: Dict[str, Circle]) -> Dict[str, List[str]]:
    result = dict()

    circle_ids = []

    for k in circles.keys():
        circle_ids.append(k)
        result[k] = []
    for i, k in enumerate(circle_ids):
        a = circles[k]

        for k2 in circle_ids[i + 1:]:
            b = circles[k2]
            d = a.distance(b)

            if d + b.radius <= a.radius + 1e-10:
                result[k2].append(k)
            elif d + a.radius <= b.radius + 1e-10:
                result[k].append(k2)
    return result


def compute_text_center(interior: List[Circle], exterior: List[Circle]):
    points = []
    for inter in interior:
        points.append(Point(inter.x, inter.y))
        points.append(Point(inter.x + inter.radius / 2, inter.y))
        points.append(Point(inter.x - inter.radius / 2, inter.y))
        points.append(Point(inter.x, inter.y + inter.radius / 2))
        points.append(Point(inter.x, inter.y - inter.radius / 2))

    initial = points[0]
    margin = initial.margin(interior, exterior)

    for pt in points[1:]:
        m = pt.margin(interior, exterior)
        if m >= margin:
            initial = pt
            margin = m

    def objective(p):
        x, y = p
        return -1 * Point(x, y).margin(interior, exterior)

    solution = minimize(objective, [initial.x, initial.y], method='nelder-mead')
    result = Point(*solution.x)

    valid = True
    for inter in interior:
        if result.distance(inter) > inter.radius:
            valid = False
            break

    for ext in exterior:
        if result.distance(ext) < ext.radius:
            valid = False
            break

    disjoint = False
    if not valid:
        if len(interior) == 1:
            result = Point(interior[0].x, interior[0].y)
        else:
            area_stats = {}
            interior[0].group_intersection_area(interior[1:], area_stats)

            arcs: List[Arc] = area_stats['arcs']
            if len(arcs) == 0:
                result = Point(0, -1000)
                disjoint = True
            elif len(arcs) == 1:
                arc = arcs[0]
                result = Point(arc.circle.x, arc.circle.y)
            elif exterior:
                result,  disjoint = compute_text_center(interior, [])
            else:
                raise ValueError("Not implemented average over all")
    return result, disjoint


def compute_text_centers(circles: Dict[str, Circle], areas: List[Overlap]):
    result = {}
    overlapped = find_overlapping_circles(circles)

    for i, area in enumerate(areas):
        area_ids = set()
        exclude = set()
        for a_j in area.sets:
            area_ids.add(a_j)

            overlaps = overlapped[a_j]
            for o in overlaps:
                exclude.add(o)

        interior = []
        exterior = []
        for k, circle in circles.items():
            if k in area_ids:
                interior.append(circle)
            elif k not in exclude:
                exterior.append(circle)

        center, disjoint = compute_text_center(interior, exterior)
        result[area.sets] = center
        if disjoint and area.size > 0:
            warnings.warn(f"Area {area} not on screen")
    return result
