from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

SMALL = 1e-10

@dataclass
class Point:
    x: float
    y: float
    label: Any = field(default=None)
    weight: float = field(default=0.0)

    def distance(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @classmethod
    def center(cls, points: List['Point']) -> 'Point':
        center = Point(0, 0)
        for pt in points:
            center.x += pt.x
            center.y += pt.y
        center.x /= len(points)
        center.y /= len(points)
        return center

    def margin(self, interior: List['Circle'], exterior: List['Circle']):
        margin = interior[0].radius - self.distance(interior[0])

        for inter in interior[1:]:
            m = inter.radius - self.distance(inter)

            if m <= margin:
                margin = m

        for ext in exterior:
            m = ext.distance(self) - ext.radius
            if m <= margin:
                margin = m
        return margin


@dataclass
class Circle(Point):
    radius: float = field(default=0.0)
    size: int = field(default=0)

    def contains(self, point: Point) -> bool:
        d = self.distance(point)
        return d <= self.radius + SMALL

    def overlap(self, other: 'Circle', d: float = None) -> float:
        if d is None:
            d = self.distance(other)

        # No overlap
        if d >= (self.radius + other.radius):
            return 0

        # Complete overlap
        if d <= abs(self.radius - other.radius):
            return np.pi * min(self.radius, other.radius) ** 2

        r12 = self.radius ** 2
        r22 = other.radius ** 2
        d2 = d ** 2

        w1: float = self.radius - (d2 - r22 + r12) / (2 * d)
        w2: float = other.radius - (d2 - r12 + r22) / (2 * d)

        return circlular_segment_area(self.radius, w1) + circlular_segment_area(other.radius, w2)

    def intersection_points(self, other: 'Circle') -> List[Point]:
        d = self.distance(other)
        r1 = self.radius
        r2 = other.radius

        # No overlap or contained-within
        if d >= (r1 + r2) or (d <= abs(r1 - r2)):
            return []

        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(r1 ** 2 - a ** 2)
        x0 = self.x + a * (other.x - self.x) / d
        y0 = self.y + a * (other.y - self.y) / d
        rx = -(other.y - self.y) * (h / d)
        ry = -(other.x - self.x) * (h / d)
        return [Point(x0 + rx, y0 - ry), Point(x0 - rx, y0 + ry)]

    @classmethod
    def group_intersection_points(cls, circles: List['Circle']) -> List[Point]:
        result = []
        for i, circle in enumerate(circles):
            for j, circle2 in enumerate(circles[i + 1:], i + 1):
                intersects = circle.intersection_points(circle2)
                for inters in intersects:
                    inters.label = [i, j]
                    result.append(inters)
        return result

    @classmethod
    def group_intersection_area(cls, circles: List['Circle'], stats: Optional[dict]=None) -> float:
        intersection_points = cls.group_intersection_points(circles)

        inner_points: List[Point] = list(
            filter(
                lambda p: all(c.contains(p) for c in circles),
                intersection_points)
        )

        arc_area: float = 0.0
        polygon_area: float = 0.0
        arcs: List = []
        i: int

        if len(inner_points) > 1:
            center = Point.center(inner_points)

            for p in inner_points:
                p.angle = np.arctan2(p.x - center.x, p.y - center.y)

            inner_points.sort(key=lambda x: x.angle)

            # Loop over all points, get the arc between points and update the
            # areas, incrementally finding the nearest points in a chain
            p2 = inner_points[-1]
            for i, p1 in enumerate(inner_points):
                polygon_area += (p2.x + p1.x) * (p1.y - p2.y)

                mid_point = Point.center([p1, p2])
                arc = None
                for parent_id in p1.label:
                    if parent_id in p2.label:
                        circle = circles[parent_id]
                        a1 = np.arctan2(p1.x - circle.x, p1.y - circle.y)
                        a2 = np.arctan2(p2.x - circle.x, p2.y - circle.y)

                        angle_diff = (a2 - a1)
                        if angle_diff < 0:
                            angle_diff += 2 * np.pi

                        a = a2 - angle_diff / 2

                        # Use the angle to find the width of the arc
                        width = mid_point.distance(
                            Point(
                                circle.x + circle.radius * np.sin(a),
                                circle.y + circle.radius * np.cos(a)
                            )
                        )

                        # clamp the width to the largest it possibly could
                        # be in case of numerical/FP issues
                        width = min(width, circle.radius * 2)

                        if arc is None or arc.width > width:
                            arc = Arc(**{
                                "circle": circle,
                                "width": width,
                                "p1": p1,
                                "p2": p2
                            })

                if arc is not None:
                    arcs.append(arc)
                    arc_area += circlular_segment_area(
                        arc.circle.radius, arc.width)
                    p2 = p1

        else:
            # No intersection points, either disjoint or completely overlapped
            smallest = min(circles, key=lambda x: x.radius)
            disjoint = False
            for circle in circles:
                if circle.distance(smallest) > abs(smallest.radius - circle.radius):
                    disjoint = True
                    break

            if disjoint:
                arc_area = polygon_area = 0.0
            else:
                arc_area = smallest.radius ** 2 * np.pi
                arcs.append(Arc(**{
                    'circle': smallest,
                    'p1': Point(smallest.x, smallest.y + smallest.radius),
                    'p2': Point(smallest.x - SMALL, smallest.y + smallest.radius),
                    'width': smallest.radius * 2
                }))

        polygon_area /= 2

        if stats is not None:
            stats['area'] = arc_area + polygon_area
            stats['arc_area'] = arc_area
            stats['polygon_area'] = polygon_area
            stats['arcs'] = arcs
            stats['inner_points'] = inner_points
            stats['intersection_points'] = intersection_points

        return arc_area + polygon_area



def circlular_segment_area(r: float, width: float) -> float:
    return r * r * np.arccos(1 - width / r) - (r - width) * np.sqrt(width * (2 * r - width))


def circle_overlap(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2

    r12 = r1 ** 2
    r22 = r2 ** 2
    d2 = d ** 2

    w1: float = r1 - (d2 - r22 + r12) / (2 * d)
    w2: float = r2 - (d2 - r12 + r22) / (2 * d)

    return circlular_segment_area(r1, w1) + circlular_segment_area(r2, w2)


@dataclass
class Arc:
    circle: Circle
    width: float
    p1: Point
    p2: Point
