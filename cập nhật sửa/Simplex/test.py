import os
import sys

# Thêm thư mục gốc chứa thư mục Simplex vào sys.path để Python nhận diện được module Simplex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Simplex.simplex_dictionary_solver import Constraint, SimplexDictionarySolver, fmt


def print_result(res):
    print("\nKẾT QUẢ TRẢ VỀ TỪ HÀM solve():")
    print("Trạng thái:", res["status"])

    if res["solution"] is not None:
        print("Nghiệm gốc:", {k: fmt(v) for k, v in res["solution"].items()})
        print("Giá trị tối ưu:", fmt(res["optimal_value"]))


def run_case_from_image():
    print("\n################ BÀI TRONG ẢNH ################")

    # max z = 2x1 - 6x2
    #
    # s.t.
    #   -x1 - x2 - x3 <= -2
    #    2x1 - x2 + x3 <= 1
    #
    #   x1, x2, x3 >= 0

    solver = SimplexDictionarySolver(
        c=[2, -6, 0],
        objective="max",
        constraints=[
            Constraint([-1, -1, -1], "<=", -2),
            Constraint([2, -1, 1], "<=", 1),
        ],
        bounds=[">=0", ">=0", ">=0"],
        var_names=["x1", "x2", "x3"],
        pivot_rule="bland",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


def run_case_with_equal_and_free_var():
    print("\n################ BÀI CÓ =, >=, BIẾN TỰ DO ################")

    # min z = x1 - x2
    #
    # s.t.
    #   x1 + x2 = 4
    #   x1 - x2 >= 2
    #   x1 <= 5
    #
    #   x1 >= 0
    #   x2 free

    solver = SimplexDictionarySolver(
        c=[1, -1],
        objective="min",
        constraints=[
            Constraint([1, 1], "=", 4),
            Constraint([1, -1], ">=", 2),
            Constraint([1, 0], "<=", 5),
        ],
        bounds=[">=0", "free"],
        var_names=["x1", "x2"],
        pivot_rule="bland",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


def run_case_without_phase1():
    print("\n################ BÀI KHÔNG CẦN PHA I ################")

    # max z = 3x1 + 2x2
    #
    # s.t.
    #   x1 + x2 <= 4
    #   x1 + 2x2 <= 6
    #
    #   x1, x2 >= 0

    solver = SimplexDictionarySolver(
        c=[3, 2],
        objective="max",
        constraints=[
            Constraint([1, 1], "<=", 4),
            Constraint([1, 2], "<=", 6),
        ],
        bounds=[">=0", ">=0"],
        var_names=["x1", "x2"],
        pivot_rule="bland",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


def run_case_dantzig():
    print("\n################ BÀI DÙNG QUY TẮC DANTZIG ################")

    solver = SimplexDictionarySolver(
        c=[3, 2],
        objective="max",
        constraints=[
            Constraint([1, 1], "<=", 4),
            Constraint([1, 2], "<=", 6),
        ],
        bounds=[">=0", ">=0"],
        var_names=["x1", "x2"],
        pivot_rule="dantzig",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


def run_case_unbounded_min():
    print("\n################ BÀI MIN KHÔNG GIỚI NỘI ################")

    # min z = -x1
    #
    # s.t.
    #   x1 >= 0
    #
    # Không có ràng buộc chặn trên x1.
    # Do đó x1 -> +infty thì z -> -infty.

    solver = SimplexDictionarySolver(
        c=[-1],
        objective="min",
        constraints=[],
        bounds=[">=0"],
        var_names=["x1"],
        pivot_rule="bland",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


def run_case_infeasible():
    print("\n################ BÀI VÔ NGHIỆM ################")

    # min z = x1
    #
    # s.t.
    #   x1 <= 1
    #   x1 >= 2
    #
    # Tương đương:
    #   x1 <= 1
    #   -x1 <= -2
    #
    # Không có nghiệm khả thi.

    solver = SimplexDictionarySolver(
        c=[1],
        objective="min",
        constraints=[
            Constraint([1], "<=", 1),
            Constraint([1], ">=", 2),
        ],
        bounds=[">=0"],
        var_names=["x1"],
        pivot_rule="bland",
        verbose=True,
    )

    res = solver.solve()
    print_result(res)


if __name__ == "__main__":
    run_case_from_image()
    run_case_with_equal_and_free_var()
    run_case_without_phase1()
    run_case_dantzig()
    run_case_unbounded_min()
    run_case_infeasible()