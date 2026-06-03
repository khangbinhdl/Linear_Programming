from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


Number = int | float | str | Fraction
Sense = Literal["<=", ">=", "="]
ObjectiveSense = Literal["min", "max"]
BoundKind = Literal[">=0", "<=0", "free"]


@dataclass
class Constraint2D:
    a: Number
    b: Number
    sense: Sense
    rhs: Number
    label: Optional[str] = None


def F(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x

    return Fraction(str(x))


def fmt(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)

    return f"{q.numerator}/{q.denominator}"


def fmt_float(x: float, digits: int = 4) -> str:
    if abs(x) < 1e-10:
        x = 0.0

    return f"{x:.{digits}g}"


def lcm_int(a: int, b: int) -> int:
    if a == 0:
        return abs(b)

    if b == 0:
        return abs(a)

    return abs(a * b) // gcd(abs(a), abs(b))


def lcm_list(values: List[int]) -> int:
    ans = 1

    for v in values:
        if v != 0:
            ans = lcm_int(ans, abs(v))

    return ans


def objective_lcm_value(c: Tuple[Number, Number]) -> Fraction:
    c1 = F(c[0])
    c2 = F(c[1])

    den_lcm = lcm_list([c1.denominator, c2.denominator])
    i1 = c1 * den_lcm
    i2 = c2 * den_lcm

    nums = [abs(i1.numerator), abs(i2.numerator)]
    z0 = lcm_list(nums)

    if z0 == 0:
        return Fraction(1)

    return Fraction(z0)


def eval_constraint(cons: Constraint2D, x1: float, x2: float) -> float:
    return float(F(cons.a)) * x1 + float(F(cons.b)) * x2


def satisfies_constraint(
    cons: Constraint2D,
    x1: float,
    x2: float,
    eps: float = 1e-9,
) -> bool:
    lhs = eval_constraint(cons, x1, x2)
    rhs = float(F(cons.rhs))

    if cons.sense == "<=":
        return lhs <= rhs + eps

    if cons.sense == ">=":
        return lhs >= rhs - eps

    if cons.sense == "=":
        return abs(lhs - rhs) <= eps

    raise ValueError(f"Dấu ràng buộc không hợp lệ: {cons.sense}")


def satisfies_all(
    constraints: List[Constraint2D],
    x1: float,
    x2: float,
    eps: float = 1e-9,
) -> bool:
    return all(satisfies_constraint(cons, x1, x2, eps) for cons in constraints)


def line_intersection(
    c1: Constraint2D,
    c2: Constraint2D,
    eps: float = 1e-12,
) -> Optional[Tuple[float, float]]:
    a1 = float(F(c1.a))
    b1 = float(F(c1.b))
    r1 = float(F(c1.rhs))

    a2 = float(F(c2.a))
    b2 = float(F(c2.b))
    r2 = float(F(c2.rhs))

    det = a1 * b2 - a2 * b1

    if abs(det) <= eps:
        return None

    x1 = (r1 * b2 - r2 * b1) / det
    x2 = (a1 * r2 - a2 * r1) / det

    return x1, x2


def add_bound_constraints(
    constraints: List[Constraint2D],
    bounds: Tuple[BoundKind, BoundKind],
) -> List[Constraint2D]:
    out = list(constraints)

    bx1, bx2 = bounds

    if bx1 == ">=0":
        out.append(Constraint2D(1, 0, ">=", 0, "x1 >= 0"))
    elif bx1 == "<=0":
        out.append(Constraint2D(1, 0, "<=", 0, "x1 <= 0"))
    elif bx1 == "free":
        pass
    else:
        raise ValueError(f"Điều kiện dấu không hợp lệ cho x1: {bx1}")

    if bx2 == ">=0":
        out.append(Constraint2D(0, 1, ">=", 0, "x2 >= 0"))
    elif bx2 == "<=0":
        out.append(Constraint2D(0, 1, "<=", 0, "x2 <= 0"))
    elif bx2 == "free":
        pass
    else:
        raise ValueError(f"Điều kiện dấu không hợp lệ cho x2: {bx2}")

    return out


def find_feasible_vertices(
    constraints: List[Constraint2D],
    eps: float = 1e-8,
) -> List[Tuple[float, float]]:
    vertices: List[Tuple[float, float]] = []

    n = len(constraints)

    for i in range(n):
        for j in range(i + 1, n):
            p = line_intersection(constraints[i], constraints[j])

            if p is None:
                continue

            x1, x2 = p

            if not np.isfinite(x1) or not np.isfinite(x2):
                continue

            if satisfies_all(constraints, x1, x2, eps):
                vertices.append((x1, x2))

    unique: List[Tuple[float, float]] = []

    for p in vertices:
        if not any(np.linalg.norm(np.array(p) - np.array(q)) <= 1e-7 for q in unique):
            unique.append(p)

    return unique


def objective_value(c: Tuple[Number, Number], x1: float, x2: float) -> float:
    return float(F(c[0])) * x1 + float(F(c[1])) * x2


def sort_polygon_vertices(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(vertices) <= 2:
        return vertices

    center = np.mean(np.array(vertices), axis=0)

    return sorted(
        vertices,
        key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]),
    )


def choose_optimal_set(
    vertices: List[Tuple[float, float]],
    c: Tuple[Number, Number],
    objective: ObjectiveSense,
    eps: float = 1e-8,
):
    if not vertices:
        return {
            "kind": "none",
            "optimal_value": None,
            "optimal_vertices": [],
        }

    scored = [(p, objective_value(c, p[0], p[1])) for p in vertices]

    if objective == "min":
        best_value = min(v for _, v in scored)
        optimal_vertices = [p for p, v in scored if abs(v - best_value) <= eps]
    elif objective == "max":
        best_value = max(v for _, v in scored)
        optimal_vertices = [p for p, v in scored if abs(v - best_value) <= eps]
    else:
        raise ValueError(f"Loại bài toán không hợp lệ: {objective}")

    if len(optimal_vertices) == 1:
        return {
            "kind": "single",
            "optimal_value": best_value,
            "optimal_vertices": optimal_vertices,
        }

    return {
        "kind": "multiple",
        "optimal_value": best_value,
        "optimal_vertices": optimal_vertices,
    }


def constraint_label(cons: Constraint2D) -> str:
    if cons.label is not None and cons.label.strip() != "":
        return cons.label

    a = F(cons.a)
    b = F(cons.b)
    rhs = F(cons.rhs)

    terms = []

    if a != 0:
        if a == 1:
            terms.append("x1")
        elif a == -1:
            terms.append("-x1")
        else:
            terms.append(f"{fmt(a)}x1")

    if b != 0:
        if b == 1:
            terms.append("x2")
        elif b == -1:
            terms.append("-x2")
        else:
            terms.append(f"{fmt(b)}x2")

    if not terms:
        lhs = "0"
    else:
        lhs = terms[0]

        for term in terms[1:]:
            if term.startswith("-"):
                lhs += f" - {term[1:]}"
            else:
                lhs += f" + {term}"

    return f"{lhs} {cons.sense} {fmt(rhs)}"


def feasible_normal_direction(cons: Constraint2D) -> Optional[np.ndarray]:
    a = float(F(cons.a))
    b = float(F(cons.b))

    n = np.array([a, b], dtype=float)
    norm = np.linalg.norm(n)

    if norm <= 1e-12:
        return None

    n = n / norm

    if cons.sense == "<=":
        return -n

    if cons.sense == ">=":
        return n

    return None


def point_on_line_near_center(
    a: float,
    b: float,
    rhs: float,
    center_x: float,
    center_y: float,
) -> Tuple[float, float]:
    normal = np.array([a, b], dtype=float)
    center = np.array([center_x, center_y], dtype=float)
    denom = float(np.dot(normal, normal))

    if denom <= 1e-12:
        return center_x, center_y

    signed_distance_factor = (np.dot(normal, center) - rhs) / denom
    p = center - signed_distance_factor * normal

    return float(p[0]), float(p[1])


def get_line_segment_in_box(
    a: float,
    b: float,
    rhs: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    points: List[Tuple[float, float]] = []

    if abs(b) > 1e-12:
        for x in [x_min, x_max]:
            y = (rhs - a * x) / b
            if y_min - 1e-9 <= y <= y_max + 1e-9:
                points.append((x, y))

    if abs(a) > 1e-12:
        for y in [y_min, y_max]:
            x = (rhs - b * y) / a
            if x_min - 1e-9 <= x <= x_max + 1e-9:
                points.append((x, y))

    unique: List[Tuple[float, float]] = []

    for p in points:
        if not any(np.linalg.norm(np.array(p) - np.array(q)) <= 1e-7 for q in unique):
            unique.append(p)

    if len(unique) < 2:
        return None

    return unique[0], unique[1]


def collect_scale_points(
    constraints: List[Constraint2D],
    c: Tuple[Number, Number],
    vertices: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = [(0.0, 0.0)]

    for p in vertices:
        if np.isfinite(p[0]) and np.isfinite(p[1]):
            points.append(p)

    n = len(constraints)

    for i in range(n):
        for j in range(i + 1, n):
            p = line_intersection(constraints[i], constraints[j])

            if p is not None and np.isfinite(p[0]) and np.isfinite(p[1]):
                points.append(p)

    for cons in constraints:
        a = float(F(cons.a))
        b = float(F(cons.b))
        rhs = float(F(cons.rhs))

        if abs(a) > 1e-12:
            points.append((rhs / a, 0.0))

        if abs(b) > 1e-12:
            points.append((0.0, rhs / b))

    c1 = float(F(c[0]))
    c2 = float(F(c[1]))
    z0 = float(objective_lcm_value(c))

    if abs(c1) > 1e-12:
        points.append((z0 / c1, 0.0))

    if abs(c2) > 1e-12:
        points.append((0.0, z0 / c2))

    clean_points = []

    for x1, x2 in points:
        if np.isfinite(x1) and np.isfinite(x2):
            clean_points.append((x1, x2))

    return clean_points


def auto_equal_limits(
    constraints: List[Constraint2D],
    c: Tuple[Number, Number],
    vertices: List[Tuple[float, float]],
    scale_factor: float = 1.35,
    min_radius: float = 5.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    points = collect_scale_points(
        constraints=constraints,
        c=c,
        vertices=vertices,
    )

    if not points:
        return (-10, 10), (-10, 10)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min_raw = min(xs)
    x_max_raw = max(xs)
    y_min_raw = min(ys)
    y_max_raw = max(ys)

    cx = 0.5 * (x_min_raw + x_max_raw)
    cy = 0.5 * (y_min_raw + y_max_raw)

    half_x = 0.5 * (x_max_raw - x_min_raw)
    half_y = 0.5 * (y_max_raw - y_min_raw)

    radius = max(half_x, half_y, min_radius)
    radius *= scale_factor

    xlim = (cx - radius, cx + radius)
    ylim = (cy - radius, cy + radius)

    return xlim, ylim


def add_constraint_trace(
    fig: go.Figure,
    cons: Constraint2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    a = float(F(cons.a))
    b = float(F(cons.b))
    rhs = float(F(cons.rhs))
    label = constraint_label(cons)

    segment = get_line_segment_in_box(a, b, rhs, x_min, x_max, y_min, y_max)

    if segment is None:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(width=2),
                showlegend=True,
                hoverinfo="skip",
            )
        )
        return

    p1, p2 = segment

    fig.add_trace(
        go.Scatter(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            mode="lines",
            name=label,
            line=dict(width=2),
            hovertemplate=f"{label}<extra></extra>",
            showlegend=True,
        )
    )


def add_constraint_normal_arrow(
    fig: go.Figure,
    cons: Constraint2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    arrow_length: float,
):
    direction = feasible_normal_direction(cons)

    if direction is None:
        return

    a = float(F(cons.a))
    b = float(F(cons.b))
    rhs = float(F(cons.rhs))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    x0, y0 = point_on_line_near_center(a, b, rhs, cx, cy)

    x1 = x0 + arrow_length * direction[0]
    y1 = y0 + arrow_length * direction[1]

    if not all(np.isfinite(v) for v in [x0, y0, x1, y1]):
        return

    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1.3,
        opacity=0.75,
        text="",
    )


def add_objective_line_and_arrow(
    fig: go.Figure,
    c: Tuple[Number, Number],
    objective: ObjectiveSense,
    z0: Fraction,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    c1 = float(F(c[0]))
    c2 = float(F(c[1]))
    z_float = float(z0)

    if abs(c1) <= 1e-12 and abs(c2) <= 1e-12:
        raise ValueError("Vector hệ số hàm mục tiêu không được đồng thời bằng 0.")

    segment = get_line_segment_in_box(c1, c2, z_float, x_min, x_max, y_min, y_max)

    label = f"Đường mục tiêu mẫu: z = {fmt(z0)}"

    if segment is not None:
        p1, p2 = segment

        fig.add_trace(
            go.Scatter(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                mode="lines",
                name=label,
                line=dict(width=2, dash="dash"),
                hovertemplate=f"{label}<extra></extra>",
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(width=2, dash="dash"),
                showlegend=True,
                hoverinfo="skip",
            )
        )

    normal = np.array([c1, c2], dtype=float)
    norm = np.linalg.norm(normal)

    if norm <= 1e-12:
        return

    normal = normal / norm

    if objective == "max":
        direction = normal
    else:
        direction = -normal

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    x0, y0 = point_on_line_near_center(c1, c2, z_float, cx, cy)

    arrow_length = 0.07 * max(x_max - x_min, y_max - y_min)

    x1 = x0 + arrow_length * direction[0]
    y1 = y0 + arrow_length * direction[1]

    if not all(np.isfinite(v) for v in [x0, y0, x1, y1]):
        return

    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        opacity=0.9,
        text="",
    )


def add_optimal_objective_line(
    fig: go.Figure,
    c: Tuple[Number, Number],
    z_star: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    c1 = float(F(c[0]))
    c2 = float(F(c[1]))

    segment = get_line_segment_in_box(c1, c2, z_star, x_min, x_max, y_min, y_max)
    label = f"Đường mục tiêu tối ưu: z = {z_star:.4g}"

    if segment is None:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(width=3),
                showlegend=True,
                hoverinfo="skip",
            )
        )
        return

    p1, p2 = segment

    fig.add_trace(
        go.Scatter(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            mode="lines",
            name=label,
            line=dict(width=3),
            hovertemplate=f"{label}<extra></extra>",
            showlegend=True,
        )
    )


def add_feasible_region(
    fig: go.Figure,
    vertices: List[Tuple[float, float]],
):
    if not vertices:
        return

    polygon = sort_polygon_vertices(vertices)

    if len(polygon) < 3:
        return

    xs = [p[0] for p in polygon] + [polygon[0][0]]
    ys = [p[1] for p in polygon] + [polygon[0][1]]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            fill="toself",
            name="Miền chấp nhận được",
            line=dict(width=1),
            opacity=0.35,
            hovertemplate="Miền chấp nhận được<extra></extra>",
            showlegend=True,
        )
    )


def add_vertices(
    fig: go.Figure,
    vertices: List[Tuple[float, float]],
):
    for i, (x1, x2) in enumerate(vertices, start=1):
        label = f"V{i} = ({fmt_float(x1)}, {fmt_float(x2)})"

        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[x2],
                mode="markers",
                name=label,
                marker=dict(size=8),
                hovertemplate=f"{label}<extra></extra>",
                showlegend=True,
            )
        )


def add_optimal_solution(
    fig: go.Figure,
    optimal_info,
):
    kind = optimal_info["kind"]
    z_star = optimal_info["optimal_value"]
    optimal_vertices = optimal_info["optimal_vertices"]

    if kind == "none":
        return

    if kind == "single":
        x_star, y_star = optimal_vertices[0]
        label = (
            f"Điểm tối ưu: "
            f"({fmt_float(x_star)}, {fmt_float(y_star)}), "
            f"z = {fmt_float(z_star)}"
        )

        fig.add_trace(
            go.Scatter(
                x=[x_star],
                y=[y_star],
                mode="markers",
                name=label,
                marker=dict(size=14, symbol="star"),
                hovertemplate=f"{label}<extra></extra>",
                showlegend=True,
            )
        )

    elif kind == "multiple":
        pts = sort_polygon_vertices(optimal_vertices)

        if len(pts) >= 2:
            p1 = pts[0]
            p2 = pts[-1]

            label = (
                "Đoạn nghiệm tối ưu: "
                f"({fmt_float(p1[0])}, {fmt_float(p1[1])})"
                " đến "
                f"({fmt_float(p2[0])}, {fmt_float(p2[1])}), "
                f"z = {fmt_float(z_star)}"
            )

            fig.add_trace(
                go.Scatter(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=5),
                    marker=dict(size=12, symbol="star"),
                    hovertemplate=f"{label}<extra></extra>",
                    showlegend=True,
                )
            )


def plot_lp_2d(
    c: Tuple[Number, Number],
    constraints: List[Constraint2D],
    objective: ObjectiveSense = "max",
    bounds: Tuple[BoundKind, BoundKind] = (">=0", ">=0"),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Giải quy hoạch tuyến tính 2 biến bằng phương pháp hình học",
    show: bool = True,
    draw_constraint_normals: bool = True,
    scale_factor: float = 1.35,
    min_radius: float = 5.0,
):
    full_constraints = add_bound_constraints(constraints, bounds)
    vertices = find_feasible_vertices(full_constraints)
    optimal_info = choose_optimal_set(vertices, c, objective)

    auto_xlim, auto_ylim = auto_equal_limits(
        constraints=full_constraints,
        c=c,
        vertices=vertices,
        scale_factor=scale_factor,
        min_radius=min_radius,
    )

    if xlim is None:
        xlim = auto_xlim

    if ylim is None:
        ylim = auto_ylim

    x_min, x_max = xlim
    y_min, y_max = ylim

    # Ép hai trục có cùng độ dài số học.
    # Plotly vẫn cho zoom/pan tương tác, nhưng ban đầu là khung vuông.
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)

    x_min = x_center - radius
    x_max = x_center + radius
    y_min = y_center - radius
    y_max = y_center + radius

    fig = go.Figure()

    for cons in full_constraints:
        add_constraint_trace(
            fig=fig,
            cons=cons,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    add_feasible_region(fig, vertices)
    add_vertices(fig, vertices)

    if draw_constraint_normals:
        arrow_length = 0.045 * max(x_max - x_min, y_max - y_min)

        for cons in full_constraints:
            add_constraint_normal_arrow(
                fig=fig,
                cons=cons,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                arrow_length=arrow_length,
            )

    z0 = objective_lcm_value(c)

    add_objective_line_and_arrow(
        fig=fig,
        c=c,
        objective=objective,
        z0=z0,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    if optimal_info["kind"] != "none":
        add_optimal_objective_line(
            fig=fig,
            c=c,
            z_star=optimal_info["optimal_value"],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        add_optimal_solution(fig, optimal_info)

    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="x1",
        yaxis_title="x2",
        width=1050,
        height=750,
        hovermode="closest",
        legend=dict(
            title="Chú thích",
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=40, r=300, t=70, b=40),
        template="plotly_white",
    )

    fig.update_xaxes(
        range=[x_min, x_max],
        zeroline=True,
        showgrid=True,
        constrain="domain",
    )

    fig.update_yaxes(
        range=[y_min, y_max],
        zeroline=True,
        showgrid=True,
        scaleanchor="x",
        scaleratio=1,
    )

    if show:
        fig.show()

    result_base = {
        "status": None,
        "vertices": vertices,
        "optimal_type": optimal_info["kind"],
        "optimal_vertices": optimal_info["optimal_vertices"],
        "optimal_value": optimal_info["optimal_value"],
        "figure": fig,
    }

    if optimal_info["kind"] == "none":
        result_base["status"] = "không tìm được đỉnh khả thi hữu hạn"
        return result_base

    if optimal_info["kind"] == "single":
        result_base["status"] = "có nghiệm tối ưu duy nhất trên các đỉnh hữu hạn tìm được"
        result_base["optimal_point"] = optimal_info["optimal_vertices"][0]
        return result_base

    result_base["status"] = "có vô số nghiệm tối ưu trên một đoạn thẳng"
    result_base["optimal_segment"] = optimal_info["optimal_vertices"]
    return result_base


if __name__ == "__main__":
    result = plot_lp_2d(
        c=(2, 3),
        objective="max",
        constraints=[
            Constraint2D(1, 1, "<=", 4),
            Constraint2D(1, 0, "<=", 2),
            Constraint2D(0, 1, "<=", 3),
        ],
        bounds=(">=0", ">=0"),
        title="Ví dụ: max z = 2x1 + 3x2",
        show=True,
        draw_constraint_normals=True,
    )

    print("Trạng thái:", result["status"])
    print("Các đỉnh khả thi:", result["vertices"])
    print("Kiểu nghiệm tối ưu:", result["optimal_type"])
    print("Nghiệm tối ưu:", result.get("optimal_point", result.get("optimal_segment")))
    print("Giá trị tối ưu:", result["optimal_value"])