import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Graphical.graphical_lp_2d import Constraint2D, plot_lp_2d


def assert_close_float(a: float, b: float, tol: float = 1e-7):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b}"


def assert_point_close(p, q, tol: float = 1e-7):
    assert len(p) == len(q)

    for pi, qi in zip(p, q):
        assert_close_float(pi, qi, tol)


def run_graphical_tests():
    print("\n\n==================== GRAPHICAL LP 2D TESTS ====================")

    os.makedirs("test_outputs", exist_ok=True)

    # ------------------------------------------------------------
    # Test 1: max cơ bản, nghiệm tối ưu duy nhất
    # max z = 2x + 3y
    # s.t.
    #   x + y <= 4
    #   x <= 2
    #   y <= 3
    #   x, y >= 0
    #
    # Optimum: (1, 3), z = 11
    # ------------------------------------------------------------

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
        title="Graphical Test 1: max z = 2x + 3y",
        save_path="test_outputs/graphical_test_1_unique_max.png",
        show=False,
    )

    print("Graphical Test 1:", result["status"])
    assert result["optimal_type"] == "single"
    assert_point_close(result["optimal_point"], (1, 3))
    assert_close_float(result["optimal_value"], 11)

    # ------------------------------------------------------------
    # Test 2: min cơ bản, nghiệm tối ưu duy nhất
    # min z = x + 4y
    # s.t.
    #   -2x + 2y <= 6
    #    x + 2y <= 4
    #    x, y >= 0
    #
    # Miền chứa gốc tọa độ, nên optimum là (0, 0), z = 0.
    # ------------------------------------------------------------

    result = plot_lp_2d(
        c=(1, 4),
        objective="min",
        constraints=[
            Constraint2D(-2, 2, "<=", 6, "-2x + 2y <= 6"),
            Constraint2D(1, 2, "<=", 4, "x + 2y <= 4"),
        ],
        bounds=(">=0", ">=0"),
        xlim=(-1, 6),
        ylim=(-1, 6),
        title="Graphical Test 2: min z = x + 4y",
        save_path="test_outputs/graphical_test_2_unique_min.png",
        show=False,
    )

    print("Graphical Test 2:", result["status"])
    assert result["optimal_type"] == "single"
    assert_point_close(result["optimal_point"], (0, 0))
    assert_close_float(result["optimal_value"], 0)

    # ------------------------------------------------------------
    # Test 3: vô số nghiệm tối ưu trên một đoạn thẳng
    #
    # min z = x + 2y
    #
    # s.t.
    #   3x + y >= 3
    #   x + 2y >= 4
    #   x - y <= 1
    #   x <= 5
    #   y <= 5
    #   x, y >= 0
    #
    # Đường x + 2y = 4 là một cạnh của miền chấp nhận được.
    # Hai đầu mút tối ưu:
    #   (0.4, 1.8)
    #   (2, 1)
    # Giá trị tối ưu:
    #   z = 4
    # ------------------------------------------------------------

    result = plot_lp_2d(
        c=(1, 2),
        objective="min",
        constraints=[
            Constraint2D(3, 1, ">=", 3, "3x + y >= 3"),
            Constraint2D(1, 2, ">=", 4, "x + 2y >= 4"),
            Constraint2D(1, -1, "<=", 1, "x - y <= 1"),
            Constraint2D(1, 0, "<=", 5, "x <= 5"),
            Constraint2D(0, 1, "<=", 5, "y <= 5"),
        ],
        bounds=(">=0", ">=0"),
        xlim=(-1, 7),
        ylim=(-1, 7),
        title="Graphical Test 3: multiple optima",
        save_path="test_outputs/graphical_test_3_multiple_optima.png",
        show=False,
    )

    print("Graphical Test 3:", result["status"])
    assert result["optimal_type"] == "multiple"
    assert_close_float(result["optimal_value"], 4)

    opt_vertices = result["optimal_vertices"]

    assert len(opt_vertices) == 2

    expected = [(0.4, 1.8), (2, 1)]

    for e in expected:
        assert any(
            abs(p[0] - e[0]) <= 1e-7 and abs(p[1] - e[1]) <= 1e-7
            for p in opt_vertices
        ), f"Không tìm thấy đỉnh tối ưu kỳ vọng: {e}"

    # ------------------------------------------------------------
    # Test 4: có ràng buộc >= và nghiệm tối ưu duy nhất
    #
    # max z = x + y
    #
    # s.t.
    #   x + y >= 2
    #   x <= 3
    #   y <= 2
    #   x, y >= 0
    #
    # Optimum: (3, 2), z = 5.
    # ------------------------------------------------------------

    result = plot_lp_2d(
        c=(1, 1),
        objective="max",
        constraints=[
            Constraint2D(1, 1, ">=", 2, "x + y >= 2"),
            Constraint2D(1, 0, "<=", 3, "x <= 3"),
            Constraint2D(0, 1, "<=", 2, "y <= 2"),
        ],
        bounds=(">=0", ">=0"),
        xlim=(-1, 5),
        ylim=(-1, 5),
        title="Graphical Test 4: >= constraint",
        save_path="test_outputs/graphical_test_4_ge_constraint.png",
        show=False,
    )

    print("Graphical Test 4:", result["status"])
    assert result["optimal_type"] == "single"
    assert_point_close(result["optimal_point"], (3, 2))
    assert_close_float(result["optimal_value"], 5)

    # ------------------------------------------------------------
    # Test 5: biến dấu âm x <= 0
    #
    # max z = -x + y
    #
    # s.t.
    #   x >= -3
    #   y <= 2
    #   y >= 0
    #   x <= 0
    #
    # Optimum: x = -3, y = 2, z = 5.
    # ------------------------------------------------------------

    result = plot_lp_2d(
        c=(-1, 1),
        objective="max",
        constraints=[
            Constraint2D(1, 0, ">=", -3, "x >= -3"),
            Constraint2D(0, 1, "<=", 2, "y <= 2"),
        ],
        bounds=("<=0", ">=0"),
        xlim=(-5, 2),
        ylim=(-1, 4),
        title="Graphical Test 5: x <= 0",
        save_path="test_outputs/graphical_test_5_negative_variable.png",
        show=False,
    )

    print("Graphical Test 5:", result["status"])
    assert result["optimal_type"] == "single"
    assert_point_close(result["optimal_point"], (-3, 2))
    assert_close_float(result["optimal_value"], 5)

    print("\nTẤT CẢ GRAPHICAL LP 2D TEST ĐÃ PASS.")


if __name__ == "__main__":
    run_graphical_tests()

    print("\n\nTẤT CẢ TEST ĐÃ PASS.")