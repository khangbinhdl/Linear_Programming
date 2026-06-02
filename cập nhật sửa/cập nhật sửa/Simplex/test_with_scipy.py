from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Literal, Optional, Tuple

import random
import numpy as np
from scipy.optimize import linprog

import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simplex.simplex_dictionary_solver import Constraint, SimplexDictionarySolver, fmt


ObjectiveSense = Literal["min", "max"]
BoundKind = Literal[">=0", "<=0", "free"]
Sense = Literal["<=", ">=", "="]


@dataclass
class LPCase:
    name: str
    c: List[float]
    objective: ObjectiveSense
    constraints: List[Constraint]
    bounds: List[BoundKind]
    var_names: List[str]
    expected_status: Optional[str] = None


def fraction_to_float(x) -> float:
    if isinstance(x, Fraction):
        return float(x)
    return float(x)


def solver_status_to_scipy_status(status: str) -> str:
    if status == "optimal":
        return "optimal"
    if status == "infeasible":
        return "infeasible"
    if status == "unbounded":
        return "unbounded"
    return "unknown"


def scipy_status_to_text(res) -> str:
    """
    SciPy linprog status:
        0: optimal
        1: iteration/time limit
        2: infeasible
        3: unbounded
        4: numerical difficulties
    """

    if res.status == 0:
        return "optimal"
    if res.status == 2:
        return "infeasible"
    if res.status == 3:
        return "unbounded"

    return f"other_status_{res.status}"


def convert_bounds_to_scipy(bounds: List[BoundKind]) -> List[Tuple[Optional[float], Optional[float]]]:
    scipy_bounds = []

    for b in bounds:
        if b == ">=0":
            scipy_bounds.append((0, None))
        elif b == "<=0":
            scipy_bounds.append((None, 0))
        elif b == "free":
            scipy_bounds.append((None, None))
        else:
            raise ValueError(f"Điều kiện dấu không hợp lệ: {b}")

    return scipy_bounds


def solve_with_scipy(case: LPCase):
    """
    SciPy luôn giải dạng:

        min c^T x

    Nếu bài toán gốc là max, đổi thành:

        min -c^T x
    """

    c = np.array(case.c, dtype=float)

    if case.objective == "max":
        c_scipy = -c
    else:
        c_scipy = c.copy()

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for cons in case.constraints:
        row = np.array([float(v) for v in cons.coeffs], dtype=float)
        rhs = float(cons.rhs)

        if cons.sense == "<=":
            A_ub.append(row)
            b_ub.append(rhs)

        elif cons.sense == ">=":
            A_ub.append(-row)
            b_ub.append(-rhs)

        elif cons.sense == "=":
            A_eq.append(row)
            b_eq.append(rhs)

        else:
            raise ValueError(f"Dấu ràng buộc không hợp lệ: {cons.sense}")

    A_ub_np = np.array(A_ub, dtype=float) if A_ub else None
    b_ub_np = np.array(b_ub, dtype=float) if b_ub else None
    A_eq_np = np.array(A_eq, dtype=float) if A_eq else None
    b_eq_np = np.array(b_eq, dtype=float) if b_eq else None

    res = linprog(
        c=c_scipy,
        A_ub=A_ub_np,
        b_ub=b_ub_np,
        A_eq=A_eq_np,
        b_eq=b_eq_np,
        bounds=convert_bounds_to_scipy(case.bounds),
        method="highs",
    )

    return res


def solve_with_our_solver(case: LPCase, pivot_rule: str = "bland", verbose: bool = False):
    solver = SimplexDictionarySolver(
        c=case.c,
        objective=case.objective,
        constraints=case.constraints,
        bounds=case.bounds,
        var_names=case.var_names,
        pivot_rule=pivot_rule,
        verbose=verbose,
    )

    return solver.solve()


def original_objective_value(case: LPCase, solution_dict) -> float:
    value = 0.0

    for ci, name in zip(case.c, case.var_names):
        value += float(ci) * fraction_to_float(solution_dict[name])

    return value


def check_solution_feasible(case: LPCase, solution_dict, tol: float = 1e-7):
    x = np.array([fraction_to_float(solution_dict[name]) for name in case.var_names], dtype=float)

    for idx, bound in enumerate(case.bounds):
        if bound == ">=0":
            assert x[idx] >= -tol, f"{case.name}: biến {case.var_names[idx]} vi phạm >= 0"
        elif bound == "<=0":
            assert x[idx] <= tol, f"{case.name}: biến {case.var_names[idx]} vi phạm <= 0"
        elif bound == "free":
            pass
        else:
            raise ValueError(f"Điều kiện dấu không hợp lệ: {bound}")

    for cons in case.constraints:
        lhs = sum(float(a) * xj for a, xj in zip(cons.coeffs, x))
        rhs = float(cons.rhs)

        if cons.sense == "<=":
            assert lhs <= rhs + tol, f"{case.name}: ràng buộc <= bị vi phạm"
        elif cons.sense == ">=":
            assert lhs >= rhs - tol, f"{case.name}: ràng buộc >= bị vi phạm"
        elif cons.sense == "=":
            assert abs(lhs - rhs) <= tol, f"{case.name}: ràng buộc = bị vi phạm"
        else:
            raise ValueError(f"Dấu ràng buộc không hợp lệ: {cons.sense}")


def compare_case(case: LPCase, pivot_rule: str = "bland", tol: float = 1e-7):
    print(f"\n========== TEST: {case.name} | rule={pivot_rule} ==========")

    our_res = solve_with_our_solver(case, pivot_rule=pivot_rule, verbose=False)
    scipy_res = solve_with_scipy(case)

    our_status = solver_status_to_scipy_status(our_res["status"])
    scipy_status = scipy_status_to_text(scipy_res)

    print("Solver status:", our_status)
    print("SciPy status: ", scipy_status)

    if case.expected_status is not None:
        assert our_status == case.expected_status, (
            f"{case.name}: solver trả status {our_status}, "
            f"kỳ vọng {case.expected_status}"
        )

    assert our_status == scipy_status, (
        f"{case.name}: status khác SciPy. "
        f"solver={our_status}, scipy={scipy_status}, scipy_message={scipy_res.message}"
    )

    if our_status == "optimal":
        assert our_res["solution"] is not None

        check_solution_feasible(case, our_res["solution"], tol=tol)

        our_obj = fraction_to_float(our_res["optimal_value"])

        if case.objective == "max":
            scipy_obj = -float(scipy_res.fun)
        else:
            scipy_obj = float(scipy_res.fun)

        obj_from_solution = original_objective_value(case, our_res["solution"])

        print("Solver objective:", our_obj)
        print("SciPy objective: ", scipy_obj)
        print("Objective from returned solution:", obj_from_solution)
        print("Solver solution:", {k: fmt(v) for k, v in our_res["solution"].items()})
        print("SciPy solution: ", scipy_res.x)

        assert abs(our_obj - obj_from_solution) <= tol, (
            f"{case.name}: giá trị objective không khớp nghiệm solver trả về"
        )

        assert abs(our_obj - scipy_obj) <= tol, (
            f"{case.name}: objective khác SciPy. "
            f"solver={our_obj}, scipy={scipy_obj}"
        )

    elif our_status == "infeasible":
        assert our_res["solution"] is None
        assert our_res["optimal_value"] is None
        print("Kết luận đúng: bài toán vô nghiệm.")

    elif our_status == "unbounded":
        assert our_res["solution"] is None
        assert our_res["optimal_value"] is None

        if case.objective == "min":
            print("Kết luận đúng: không giới nội, min z = -infty.")
        else:
            print("Kết luận đúng: không giới nội, max z = +infty.")

    print("PASSED")


def deterministic_cases() -> List[LPCase]:
    return [
        LPCase(
            name="max_co_ban_khong_can_pha_1",
            c=[3, 2],
            objective="max",
            constraints=[
                Constraint([1, 1], "<=", 4),
                Constraint([1, 2], "<=", 6),
            ],
            bounds=[">=0", ">=0"],
            var_names=["x1", "x2"],
            expected_status="optimal",
        ),
        LPCase(
            name="min_co_dau_bang_dau_lon_hon_va_bien_tu_do",
            c=[1, -1],
            objective="min",
            constraints=[
                Constraint([1, 1], "=", 4),
                Constraint([1, -1], ">=", 2),
                Constraint([1, 0], "<=", 5),
            ],
            bounds=[">=0", "free"],
            var_names=["x1", "x2"],
            expected_status="optimal",
        ),
        LPCase(
            name="bai_trong_anh_can_pha_1",
            c=[2, -6, 0],
            objective="max",
            constraints=[
                Constraint([-1, -1, -1], "<=", -2),
                Constraint([2, -1, 1], "<=", 1),
            ],
            bounds=[">=0", ">=0", ">=0"],
            var_names=["x1", "x2", "x3"],
            expected_status="optimal",
        ),
        LPCase(
            name="bien_am_x_le_0",
            c=[2, 1],
            objective="min",
            constraints=[
                Constraint([1, 1], ">=", -3),
                Constraint([1, -1], "<=", 2),
                Constraint([0, 1], "<=", 4),
            ],
            bounds=["<=0", ">=0"],
            var_names=["x1", "x2"],
            expected_status="optimal",
        ),
        LPCase(
            name="nhieu_nghiem_toi_uu",
            c=[1, 1],
            objective="max",
            constraints=[
                Constraint([1, 1], "<=", 4),
                Constraint([1, 0], "<=", 3),
                Constraint([0, 1], "<=", 3),
            ],
            bounds=[">=0", ">=0"],
            var_names=["x1", "x2"],
            expected_status="optimal",
        ),
        LPCase(
            name="vo_nghiem_don_gian",
            c=[1],
            objective="min",
            constraints=[
                Constraint([1], "<=", 1),
                Constraint([1], ">=", 2),
            ],
            bounds=[">=0"],
            var_names=["x1"],
            expected_status="infeasible",
        ),
        LPCase(
            name="min_khong_gioi_noi",
            c=[-1],
            objective="min",
            constraints=[],
            bounds=[">=0"],
            var_names=["x1"],
            expected_status="unbounded",
        ),
        LPCase(
            name="max_khong_gioi_noi",
            c=[1],
            objective="max",
            constraints=[],
            bounds=[">=0"],
            var_names=["x1"],
            expected_status="unbounded",
        ),
        LPCase(
            name="rang_buoc_dang_bang_tao_hai_bat_dang_thuc",
            c=[1, 2, -1],
            objective="min",
            constraints=[
                Constraint([1, 1, 1], "=", 3),
                Constraint([1, -1, 0], "<=", 1),
                Constraint([0, 1, -1], ">=", -2),
            ],
            bounds=[">=0", ">=0", "free"],
            var_names=["x1", "x2", "x3"],
            expected_status="optimal",
        ),
        LPCase(
            name="max_voi_bien_tu_do",
            c=[2, -3],
            objective="max",
            constraints=[
                Constraint([1, 1], "<=", 5),
                Constraint([-1, 2], "<=", 4),
                Constraint([1, 0], ">=", -2),
            ],
            bounds=["free", ">=0"],
            var_names=["x1", "x2"],
            expected_status="optimal",
        ),
    ]


def random_bounded_case(seed: int, n_vars: int = 3, n_constraints: int = 5) -> LPCase:
    """
    Tạo bài toán ngẫu nhiên nhưng bị chặn bằng box:

        -5 <= x_i <= 5

    Vì solver hiện tại không nhận trực tiếp lower/upper bất kỳ,
    ta đặt biến free rồi thêm ràng buộc:

        x_i <= 5
        x_i >= -5

    Như vậy bài toán luôn bị chặn.
    """

    rng = random.Random(seed)

    objective = "min" if seed % 2 == 0 else "max"

    c = [rng.randint(-5, 5) for _ in range(n_vars)]

    if all(v == 0 for v in c):
        c[0] = 1

    constraints: List[Constraint] = []

    # Box constraints: -5 <= x_i <= 5
    for j in range(n_vars):
        row_upper = [0] * n_vars
        row_upper[j] = 1
        constraints.append(Constraint(row_upper, "<=", 5))

        row_lower = [0] * n_vars
        row_lower[j] = 1
        constraints.append(Constraint(row_lower, ">=", -5))

    for _ in range(n_constraints):
        row = [rng.randint(-4, 4) for _ in range(n_vars)]

        if all(v == 0 for v in row):
            row[0] = 1

        # Chọn RHS sao cho x = 0 luôn khả thi phần lớn trường hợp.
        # Với <=: rhs >= 0
        # Với >=: rhs <= 0
        # Với = thì dùng rhs = 0 để không dễ vô nghiệm.
        sense = rng.choice(["<=", ">=", "="])

        if sense == "<=":
            rhs = rng.randint(0, 8)
        elif sense == ">=":
            rhs = rng.randint(-8, 0)
        else:
            rhs = 0

        constraints.append(Constraint(row, sense, rhs))

    return LPCase(
        name=f"random_bounded_seed_{seed}",
        c=c,
        objective=objective,
        constraints=constraints,
        bounds=["free"] * n_vars,
        var_names=[f"x{i + 1}" for i in range(n_vars)],
        expected_status=None,
    )


def run_deterministic_tests():
    print("\n\n==================== DETERMINISTIC TESTS ====================")

    for case in deterministic_cases():
        compare_case(case, pivot_rule="bland")
        compare_case(case, pivot_rule="dantzig")


def run_random_tests(num_cases: int = 30):
    print("\n\n==================== RANDOM BOUNDED TESTS ====================")

    for seed in range(num_cases):
        case = random_bounded_case(seed=seed, n_vars=3, n_constraints=5)

        # Với random test, chỉ cần Bland là đủ.
        # Nếu muốn kiểm tra thêm Dantzig thì mở dòng thứ hai.
        compare_case(case, pivot_rule="bland")
        compare_case(case, pivot_rule="dantzig")


def main():
    run_deterministic_tests()
    run_random_tests(num_cases=30)

    print("\n\nTẤT CẢ TEST ĐÃ PASS.")


if __name__ == "__main__":
    main()