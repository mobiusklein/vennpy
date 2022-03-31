import itertools
from turtle import circle

from scipy.optimize import bisect, minimize

from  typing import Any, Iterable, Optional, List, Set, Tuple, DefaultDict, Dict, FrozenSet, Union
from dataclasses import dataclass, field

import numpy as np

from .circle import Point, Circle, Arc, SMALL, circle_overlap


@dataclass(frozen=True)
class Overlap:
    sets: FrozenSet[str]
    size: int


@dataclass
class Group:
    members: Set
    label: str

    def __len__(self):
        return len(self.members)

    def copy(self) -> "Group":
        return self.__class__(self.members, self.label)

    def overlaps(self, others: List['Group']) -> Overlap:
        current_set = self.members
        labels = set()
        labels.add(self.label)
        for other in others:
            current_set = current_set & other.members
            labels.add(other.label)
        labels = frozenset(labels)
        return Overlap(labels, len(current_set))


def distance_from_intersection(r1: float, r2: float, overlap: float) -> float:
    min_r = min(r1, r2)
    if min_r ** 2 * np.pi <= overlap + SMALL:
        return abs(r1 - r2)
    return bisect(lambda distance: circle_overlap(r1, r2, distance) - overlap, 0, r1 + r2)


def create_overlaps(groups: List[Group]) -> List[Overlap]:
    overlaps = []
    for pairs in itertools.combinations(groups, 2):
        a, b = pairs
        overlap = a.overlaps([b])
        overlaps.append(overlap)
    return overlaps


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


def refine_layout(circles: Dict[str, Circle], areas: List[Overlap]):
    set_ids = []
    initials = []
    for set_id, circle in circles.items():
        initials.append(circle.x)
        initials.append(circle.y)
        set_ids.append(set_id)

    def objective(values):
        current = {}
        for i, set_id in enumerate(set_ids):
            current[set_id] = Circle(
                values[2 * i], values[2 * i + 1], set_id, circles[set_id].radius)
        return loss_function(current, areas)

    solution = minimize(objective, initials, method='nelder-mead')
    selected_coordinates = solution.x

    for i, set_id in enumerate(set_ids):
        circles[set_id].x = selected_coordinates[2 * i]
        circles[set_id].y = selected_coordinates[2 * i + 1]


class LayoutBase:
    groups: List[Group]
    circles: Dict[str, Circle]

    areas: List[Overlap]
    set_overlaps: DefaultDict[FrozenSet[str], List[Overlap]]
    scale: float = 1.0

    def _create_circles(self):
        self.circles = {}
        for group in self.groups:
            self.circles[group.label] = Circle(
                1e10,
                1e10,
                group.label,
                np.sqrt(len(group) / np.pi) * self.scale,
                len(group)
            )

    def _create_overlaps(self) -> DefaultDict[FrozenSet[str], List[Overlap]]:
        self.areas = create_overlaps(self.groups)
        set_overlaps = DefaultDict(list)

        for o in self.areas:
            left, right = o.sets
            size = o.size
            set_overlaps[left].append(
                Overlap(right, size)
            )
            set_overlaps[right].append(
                Overlap(left, size)
            )
        self.set_overlaps = set_overlaps

    def __getitem__(self, key):
        return self.circles[key]

    def __iter__(self):
        return iter(self.circles)

    def keys(self):
        return self.circles.keys()

    def values(self):
        return self.circles.values()

    def items(self):
        return self.circles.items()

    def __len__(self):
        return len(self.circles)

    def __contains__(self, key):
        return key in self.circles

    def bounding_box(self):
        return bounding_box(self.circles.values())

    def disjoint_clusters(self, flatten=True):
        clusters = DefaultDict(set, {circle.label: {circle.label} for circle in self.values()})
        for area in self.areas:
            if area.size > 0:
                new_group = set(area.sets)
                for set_id in area.sets:
                    new_group.update(clusters[set_id])
                for set_id in new_group:
                    clusters[set_id] = new_group
        if flatten:
            return list({frozenset(v) for v in clusters.values()})
        return clusters

    def layout(self):
        raise NotImplementedError()

    def refine_layout(self):
        set_ids = []
        initials = []
        for set_id, circle in self.circles.items():
            initials.append(circle.x)
            initials.append(circle.y)
            set_ids.append(set_id)

        def objective(values):
            current = {}
            for i, set_id in enumerate(set_ids):
                current[set_id] = Circle(
                    values[2 * i], values[2 * i + 1], set_id, self.circles[set_id].radius)
            return loss_function(current, self.areas)

        solution = minimize(objective, initials, method='nelder-mead')
        selected_coordinates = solution.x

        for i, set_id in enumerate(set_ids):
            self.circles[set_id].x = selected_coordinates[2 * i]
            self.circles[set_id].y = selected_coordinates[2 * i + 1]


def bounding_box(circles: Iterable[Circle]) -> Dict[str, List[float]]:
    INF = float('inf')
    xs = [INF, -INF]
    ys = [INF, -INF]

    for circle in circles:
        xs[0] = min(circle.x - circle.radius, xs[0])
        xs[1] = max(circle.x + circle.radius, xs[1])
        ys[0] = min(circle.y - circle.radius, ys[0])
        ys[1] = max(circle.y + circle.radius, ys[1])
    return {'x': xs, 'y': ys}


class GreedyLayout(LayoutBase):
    groups: List[Group]
    circles: Dict[str, Circle]

    areas: List[Overlap]
    set_overlaps: DefaultDict[FrozenSet[str], List[Overlap]]

    def __init__(self, groups: List[Group], scale: float=1.0):
        self.groups = groups
        self.scale = scale

        self._create_circles()
        self._create_overlaps()

        self.layout()
        self.refine_layout()

    def _position_circle_from_points(self, set_id, points):
        best_loss = 1e50
        best_point = points[0]
        for point in points:
            self.circles[set_id].x = point.x
            self.circles[set_id].y = point.y
            local_loss = loss_function(self.circles, self.areas)
            if local_loss < best_loss:
                best_loss = local_loss
                best_point = point
        return best_point

    def layout(self):
        most_overlapped = []
        for set_name, overlaps in self.set_overlaps.items():
            size = sum(o.size for o in overlaps)
            most_overlapped.append(
                Overlap(set_name, size)
            )
        most_overlapped.sort(key=lambda x: x.size, reverse=True)

        positioned: Dict[str, bool] = {}

        def set_position(key, x, y):
            positioned[key] = True
            self.circles[key].x = x
            self.circles[key].y = y

        most = most_overlapped[0]
        key = most.sets

        set_position(key, 0.0, 0.0)

        for seed in most_overlapped[1:]:
            set_id = seed.sets

            overlap = [o for o in self.set_overlaps[set_id] if o.sets in positioned]
            circle: Circle = self.circles[set_id]
            overlap.sort(key=lambda x: x.size, reverse=True)

            points = []
            for j, o in enumerate(overlap):
                # get appropriate distance from most overlapped already added set
                p1: Circle = self.circles[o.sets]
                d1 = distance_from_intersection(circle.radius, p1.radius, o.size)

                points.append(Point(p1.x + d1, p1.y))
                points.append(Point(p1.x - d1, p1.y))
                points.append(Point(p1.x, p1.y + d1))
                points.append(Point(p1.x, p1.y - d1))

                for k, o2 in enumerate(overlap[j + 1:], j + 1):
                    p2 = self.circles[o2.sets]
                    d2 = distance_from_intersection(circle.radius, p2.radius, o2.size)
                    points.extend(
                        Circle(p1.x, p1.y, None, d1).intersection_points(
                            Circle(p2.x, p2.y, None, d2)
                        )
                    )

            best_point = self._position_circle_from_points(set_id, points)
            set_position(set_id, best_point.x, best_point.y)


def circle_margin(current: Point, interior: List[Circle], exterior: List[Circle]):
    margin = float("inf")
    for i in interior:
        m = i.radius - i.distance(current)
        if m < margin:
            margin = m
    for i in exterior:
        m = i.distance(current) - i.radius
        if m < margin:
            margin = m
    return m


def compute_text_center(interior: List[Circle], exterior: List[Circle]):
    points = []
    for c in interior:
        points.append(Point(c.x, c.y))
        points.append(Point(c.x + c.radius / 2, c.y))
        points.append(Point(c.x - c.radius / 2, c.y))
        points.append(Point(c.x, c.y + c.radius / 2))
        points.append(Point(c.x, c.y - c.radius / 2))

    initial = points[0]
    margin = circle_margin(initial, interior, exterior)

    for point in points:
        m = circle_margin(point, interior, exterior)
        if m >= margin:
            initial = point
            margin = m

    solution = minimize(
        lambda p: circle_margin(Point(p[0], p[1]), interior, exterior),
        [initial.x, initial.y],
    ).x

    ret = Point(solution[0], solution[1])

    valid = True
    for i in interior:
        if ret.distance(i) > i.radius:
            valid = False
            break

    for i in exterior:
        if ret.distance(i) < i.radius:
            valid = False
            break

    if valid:
        return ret, False

    area_stats = {}
    Circle.group_intersection_area(interior, area_stats)
    arcs = area_stats['arcs']

    n_arcs = len(arcs)
    if n_arcs == 0:
        return None, True

    if n_arcs == 1:
        return Point(arcs[0].circle.x, arcs[0].circle.y), False

    if exterior:
        return compute_text_center(interior, [])
    return Point.center([a.p1 for a in arcs]), False


def compute_text_centers(self: LayoutBase, areas: Optional[List[Overlap]]=None):
    if areas is None:
        areas = [g.overlaps([]) for g in self.groups]
    ret = {}
    overlapped = self.disjoint_clusters(False)
    for area in areas:
        if isinstance(area.sets, str):
            area = [area.sets]
        else:
            area = list(area.sets)
        area_ids = {}
        exclude = {}

        for a in area:
            area_ids[a] = True
            overlaps = overlapped[a]

            for k in overlaps:
                exclude[k] = True

        interior = []
        exterior = []
        for set_id, circle in self.circles.items():
            if set_id in area_ids:
                interior.append(circle)
            elif set_id not in exclude:
                exterior.append(circle)

        center, disjoint = compute_text_center(interior, exterior)
        if disjoint:
            continue
        ret[frozenset(area)] = center
    return ret
