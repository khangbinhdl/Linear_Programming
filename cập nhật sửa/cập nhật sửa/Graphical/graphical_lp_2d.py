from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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

    s = f"{x:.{digits}g}"
    return s


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
    """
    Tạo giá trị z mẫu:

        z = |lcm(c1, c2)|

    Nếu c có phân số thì quy đồng trước rồi lấy lcm của tử số.
    """

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


def eval_constraint(cons: Constraint2D, x: float, y: float) -> float:
    return float(F(cons.a)) * x + float(F(cons.b)) * y


def satisfies_constraint(
    cons: Constraint2D,
    x: float,
    y: float,
    eps: float = 1e-9,
) -> bool:
    lhs = eval_constraint(cons, x, y)
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
    x: float,
    y: float,
    eps: float = 1e-9,
) -> bool:
    return all(satisfies_constraint(cons, x, y, eps) for cons in constraints)


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

    x = (r1 * b2 - r2 * b1) / det
    y = (a1 * r2 - a2 * r1) / det

    return x, y


def add_bound_constraints(
    constraints: List[Constraint2D],
    bounds: Tuple[BoundKind, BoundKind],
) -> List[Constraint2D]:
    out = list(constraints)

    bx, by = bounds

    if bx == ">=0":
        out.append(Constraint2D(1, 0, ">=", 0, "x >= 0"))
    elif bx == "<=0":
        out.append(Constraint2D(1, 0, "<=", 0, "x <= 0"))
    elif bx == "free":
        pass
    else:
        raise ValueError(f"Điều kiện dấu không hợp lệ cho x: {bx}")

    if by == ">=0":
        out.append(Constraint2D(0, 1, ">=", 0, "y >= 0"))
    elif by == "<=0":
        out.append(Constraint2D(0, 1, "<=", 0, "y <= 0"))
    elif by == "free":
        pass
    else:
        raise ValueError(f"Điều kiện dấu không hợp lệ cho y: {by}")

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

            x, y = p

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            if satisfies_all(constraints, x, y, eps):
                vertices.append((x, y))

    unique: List[Tuple[float, float]] = []

    for p in vertices:
        if not any(np.linalg.norm(np.array(p) - np.array(q)) <= 1e-7 for q in unique):
            unique.append(p)

    return unique


def objective_value(c: Tuple[Number, Number], x: float, y: float) -> float:
    return float(F(c[0])) * x + float(F(c[1])) * y


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
    """
    Tìm nghiệm tối ưu trên các đỉnh hữu hạn.

    Nếu chỉ có một đỉnh tối ưu:
        return kind = "single"

    Nếu có ít nhất hai đỉnh tối ưu có cùng giá trị z:
        return kind = "multiple"

    Khi đó toàn bộ đoạn nối giữa hai đỉnh tối ưu kề nhau trên biên
    cũng là nghiệm tối ưu.
    """

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
    if cons.label is not None:
        return cons.label

    return f"{fmt(F(cons.a))}x + {fmt(F(cons.b))}y {cons.sense} {fmt(F(cons.rhs))}"


def point_on_line_near_center(
    a: float,
    b: float,
    rhs: float,
    center_x: float,
    center_y: float,
) -> Tuple[float, float]:
    """
    Tìm điểm trên đường thẳng:

        ax + by = rhs

    gần tâm hình vẽ nhất.
    """

    normal = np.array([a, b], dtype=float)
    center = np.array([center_x, center_y], dtype=float)
    denom = float(np.dot(normal, normal))

    if denom <= 1e-12:
        return center_x, center_y

    signed_distance_factor = (np.dot(normal, center) - rhs) / denom
    p = center - signed_distance_factor * normal

    return float(p[0]), float(p[1])


def feasible_normal_direction(cons: Constraint2D) -> Optional[np.ndarray]:
    """
    Vector pháp tuyến chỉ về phía miền chấp nhận được.

    Với:
        ax + by <= c

    miền chấp nhận được nằm theo hướng:

        -(a, b)

    Với:
        ax + by >= c

    miền chấp nhận được nằm theo hướng:

        +(a, b)

    Với:
        ax + by = c

    không có một phía duy nhất.
    """

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


def draw_arrow_segment(
    ax,
    start: Tuple[float, float],
    direction: np.ndarray,
    length: float,
    label: Optional[str] = None,
    linewidth: float = 1.1,
    mutation_scale: int = 8,
    alpha: float = 0.85,
):
    """
    Vẽ mũi tên bằng điểm đầu và vector hướng.
    Dùng annotate để tránh lỗi hình học khi dùng quiver.
    """

    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)

    if norm <= 1e-12:
        return

    d = d / norm

    x0, y0 = start
    x1 = x0 + length * d[0]
    y1 = y0 + length * d[1]

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
        ),
    )

    if label is not None:
        ax.text(x1, y1, label)


def draw_constraint_line(
    ax,
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

    xs = np.linspace(x_min, x_max, 600)

    if abs(b) > 1e-12:
        ys = (rhs - a * xs) / b
        mask = (ys >= y_min - 1) & (ys <= y_max + 1)

        ax.plot(xs[mask], ys[mask], linewidth=1.1, label=label)
    elif abs(a) > 1e-12:
        x0 = rhs / a
        ax.axvline(x0, linewidth=1.1, label=label)


def draw_constraint_normal(
    ax,
    cons: Constraint2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    arrow_length: float,
):
    """
    Vẽ vector pháp tuyến của ràng buộc.
    Không đưa vector pháp tuyến vào legend.
    """

    direction = feasible_normal_direction(cons)

    if direction is None:
        return

    a = float(F(cons.a))
    b = float(F(cons.b))
    rhs = float(F(cons.rhs))

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    px, py = point_on_line_near_center(a, b, rhs, cx, cy)

    draw_arrow_segment(
        ax=ax,
        start=(px, py),
        direction=direction,
        length=arrow_length,
        label=None,
        linewidth=0.9,
        mutation_scale=7,
        alpha=0.65,
    )


def draw_objective_line_and_normal(
    ax,
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

    xs = np.linspace(x_min, x_max, 600)

    if abs(c2) > 1e-12:
        ys = (z_float - c1 * xs) / c2
        mask = (ys >= y_min - 1) & (ys <= y_max + 1)

        ax.plot(
            xs[mask],
            ys[mask],
            linestyle="--",
            linewidth=1.1,
            label=f"Đường mục tiêu mẫu: z = {fmt(z0)}",
        )
    elif abs(c1) > 1e-12:
        x0 = z_float / c1

        ax.axvline(
            x0,
            linestyle="--",
            linewidth=1.1,
            label=f"Đường mục tiêu mẫu: z = {fmt(z0)}",
        )
    else:
        raise ValueError("Vector hệ số hàm mục tiêu không được đồng thời bằng 0.")

    # Đường mục tiêu:
    #     c1*x + c2*y = z0
    #
    # Vector pháp tuyến đúng:
    #     n = (c1, c2)
    normal = np.array([c1, c2], dtype=float)
    norm = np.linalg.norm(normal)

    if norm <= 1e-12:
        return

    normal = normal / norm

    if objective == "max":
        direction = normal
        direction_text = "hướng tăng z"
    else:
        direction = -normal
        direction_text = "hướng giảm z"

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    px, py = point_on_line_near_center(c1, c2, z_float, cx, cy)

    arrow_length = 0.06 * max(x_max - x_min, y_max - y_min)

    draw_arrow_segment(
        ax=ax,
        start=(px, py),
        direction=direction,
        length=arrow_length,
        label=direction_text,
        linewidth=1.3,
        mutation_scale=9,
        alpha=0.9,
    )


def draw_optimal_objective_line(
    ax,
    c: Tuple[Number, Number],
    z_star: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    c1 = float(F(c[0]))
    c2 = float(F(c[1]))

    xs = np.linspace(x_min, x_max, 600)

    if abs(c2) > 1e-12:
        ys = (z_star - c1 * xs) / c2
        mask = (ys >= y_min - 1) & (ys <= y_max + 1)

        ax.plot(
            xs[mask],
            ys[mask],
            linewidth=1.4,
            label=f"Đường mục tiêu tối ưu: z = {z_star:.4g}",
        )
    elif abs(c1) > 1e-12:
        x0 = z_star / c1

        ax.axvline(
            x0,
            linewidth=1.4,
            label=f"Đường mục tiêu tối ưu: z = {z_star:.4g}",
        )


def draw_optimal_solution(
    ax,
    optimal_info,
):
    kind = optimal_info["kind"]
    z_star = optimal_info["optimal_value"]
    optimal_vertices = optimal_info["optimal_vertices"]

    if kind == "none":
        return

    if kind == "single":
        x_star, y_star = optimal_vertices[0]

        ax.scatter(
            [x_star],
            [y_star],
            s=90,
            marker="*",
            zorder=10,
            label=(
                f"Điểm tối ưu: "
                f"({fmt_float(x_star)}, {fmt_float(y_star)}), "
                f"z = {fmt_float(z_star)}"
            ),
        )

    elif kind == "multiple":
        pts = sort_polygon_vertices(optimal_vertices)

        if len(pts) >= 2:
            p1 = pts[0]
            p2 = pts[-1]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linewidth=3.0,
                solid_capstyle="round",
                label=(
                    "Đoạn nghiệm tối ưu: "
                    f"({fmt_float(p1[0])}, {fmt_float(p1[1])})"
                    " đến "
                    f"({fmt_float(p2[0])}, {fmt_float(p2[1])}), "
                    f"z = {fmt_float(z_star)}"
                ),
            )

            ax.scatter(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                s=90,
                marker="*",
                zorder=10,
                label="Hai đầu mút của đoạn nghiệm tối ưu",
            )


def add_vertices_to_plot(
    ax,
    vertices: List[Tuple[float, float]],
):
    """
    Đánh dấu các đỉnh của miền chấp nhận được.
    Tọa độ từng đỉnh được đưa vào legend bên ngoài hình.
    """

    if not vertices:
        return

    for i, (x, y) in enumerate(vertices, start=1):
        ax.scatter(
            [x],
            [y],
            s=32,
            zorder=6,
            label=f"V{i} = ({fmt_float(x)}, {fmt_float(y)})",
        )


def remove_duplicate_legend_entries(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}

    for h, label in zip(handles, labels):
        if label not in unique:
            unique[label] = h

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="best",
        fontsize=8.5,
    )

def place_legend_in_separate_axis(fig, legend_ax, plot_ax):
    """
    Đặt legend trong một vùng riêng bên phải đồ thị.
    Loại bỏ các mục không cần thiết khỏi legend.
    """

    handles, labels = plot_ax.get_legend_handles_labels()

    skip_labels = {
        "Miền chấp nhận được",
    }

    unique = {}

    for h, label in zip(handles, labels):
        if label in skip_labels:
            continue

        if label not in unique:
            unique[label] = h

    legend_ax.axis("off")

    legend_ax.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        fontsize=9,
        frameon=True,
    )

def plot_lp_2d(
    c: Tuple[Number, Number],
    constraints: List[Constraint2D],
    objective: ObjectiveSense = "max",
    bounds: Tuple[BoundKind, BoundKind] = (">=0", ">=0"),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Giải quy hoạch tuyến tính 2 biến bằng phương pháp hình học",
    save_path: Optional[str] = None,
    show: bool = True,
    draw_constraint_normals: bool = True,
):
    """
    Vẽ bài toán quy hoạch tuyến tính 2 biến bằng phương pháp hình học.

    Legend được đặt ở một vùng riêng bên phải đồ thị.
    """

    full_constraints = add_bound_constraints(constraints, bounds)
    vertices = find_feasible_vertices(full_constraints)
    optimal_info = choose_optimal_set(vertices, c, objective)

    if xlim is None or ylim is None:
        if vertices:
            xs = [p[0] for p in vertices]
            ys = [p[1] for p in vertices]

            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)

            if x_span <= 1e-9:
                x_span = 4

            if y_span <= 1e-9:
                y_span = 4

            x_pad = max(1, 0.35 * x_span)
            y_pad = max(1, 0.35 * y_span)

            auto_xlim = (min(xs) - x_pad, max(xs) + x_pad)
            auto_ylim = (min(ys) - y_pad, max(ys) + y_pad)
        else:
            auto_xlim = (-5, 10)
            auto_ylim = (-5, 10)

        if xlim is None:
            xlim = auto_xlim

        if ylim is None:
            ylim = auto_ylim

    x_min, x_max = xlim
    y_min, y_max = ylim

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [4.5, 1.7]},
    )

    arrow_length = 0.028 * max(x_max - x_min, y_max - y_min)

    for cons in full_constraints:
        draw_constraint_line(ax, cons, x_min, x_max, y_min, y_max)

    if draw_constraint_normals:
        for cons in full_constraints:
            draw_constraint_normal(
                ax=ax,
                cons=cons,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                arrow_length=arrow_length,
            )

    if vertices:
        polygon = sort_polygon_vertices(vertices)

        if len(polygon) >= 3:
            poly_arr = np.array(polygon)

            ax.fill(
                poly_arr[:, 0],
                poly_arr[:, 1],
                alpha=0.22,
                label="Miền chấp nhận được",
            )

        add_vertices_to_plot(ax, vertices)

    z0 = objective_lcm_value(c)

    draw_objective_line_and_normal(
        ax=ax,
        c=c,
        objective=objective,
        z0=z0,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    if optimal_info["kind"] != "none":
        draw_optimal_objective_line(
            ax=ax,
            c=c,
            z_star=optimal_info["optimal_value"],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        draw_optimal_solution(ax, optimal_info)

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)

    ax.set_aspect("equal", adjustable="box")

    place_legend_in_separate_axis(fig, legend_ax, ax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    if optimal_info["kind"] == "none":
        return {
            "status": "không tìm được đỉnh khả thi hữu hạn",
            "vertices": vertices,
            "optimal_type": "none",
            "optimal_vertices": [],
            "optimal_value": None,
            "figure": fig,
            "axis": ax,
            "legend_axis": legend_ax,
        }

    if optimal_info["kind"] == "single":
        return {
            "status": "có nghiệm tối ưu duy nhất trên các đỉnh hữu hạn tìm được",
            "vertices": vertices,
            "optimal_type": "single",
            "optimal_vertices": optimal_info["optimal_vertices"],
            "optimal_point": optimal_info["optimal_vertices"][0],
            "optimal_value": optimal_info["optimal_value"],
            "figure": fig,
            "axis": ax,
            "legend_axis": legend_ax,
        }

    return {
        "status": "có vô số nghiệm tối ưu trên một đoạn thẳng",
        "vertices": vertices,
        "optimal_type": "multiple",
        "optimal_vertices": optimal_info["optimal_vertices"],
        "optimal_segment": optimal_info["optimal_vertices"],
        "optimal_value": optimal_info["optimal_value"],
        "figure": fig,
        "axis": ax,
        "legend_axis": legend_ax,
    }


if __name__ == "__main__":
    # Ví dụ 1:
    #
    # max z = 2x + 3y
    #
    # s.t.
    #   x + y <= 4
    #   x <= 2
    #   y <= 3
    #   x >= 0
    #   y >= 0

    result = plot_lp_2d(
        c=(2, 3),
        objective="max",
        constraints=[
            Constraint2D(1, 1, "<=", 4, "x + y <= 4"),
            Constraint2D(1, 0, "<=", 2, "x <= 2"),
            Constraint2D(0, 1, "<=", 3, "y <= 3"),
        ],
        bounds=(">=0", ">=0"),
        xlim=(-1, 6),
        ylim=(-1, 6),
        title="Ví dụ: max z = 2x + 3y",
        save_path="lp_2d_example.png",
        show=True,
    )

    print("Trạng thái:", result["status"])
    print("Các đỉnh khả thi:", result["vertices"])
    print("Kiểu nghiệm tối ưu:", result["optimal_type"])
    print("Nghiệm tối ưu:", result.get("optimal_point", result.get("optimal_segment")))
    print("Giá trị tối ưu:", result["optimal_value"])