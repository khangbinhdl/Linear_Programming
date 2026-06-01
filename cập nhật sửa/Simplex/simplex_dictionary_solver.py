from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple, Optional, Literal


Number = int | float | str | Fraction
Sense = Literal["<=", ">=", "="]
ObjectiveSense = Literal["min", "max"]
BoundKind = Literal[">=0", "<=0", "free"]
Rule = Literal["bland", "dantzig"]


def F(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(str(x))


def fmt(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def clean_dict(d: Dict[str, Fraction]) -> Dict[str, Fraction]:
    return {k: v for k, v in d.items() if v != 0}


@dataclass
class Constraint:
    coeffs: List[Number]
    sense: Sense
    rhs: Number


@dataclass
class CanonicalLP:
    """
    Dạng chuẩn nội bộ:

        min c^T y

    với:

        A y <= b
        y >= 0
    """

    c_min: Dict[str, Fraction]
    A_le: List[Dict[str, Fraction]]
    b_le: List[Fraction]
    var_order: List[str]
    original_names: List[str]
    original_expr: Dict[str, List[Tuple[Fraction, str]]]
    original_sense: ObjectiveSense


@dataclass
class Row:
    """
    Một dòng từ vựng:

        biến_cơ_sở = hằng_số + tổng hệ_số * biến_không_cơ_sở
    """

    const: Fraction
    coeffs: Dict[str, Fraction]


class SimplexDictionarySolver:
    def __init__(
        self,
        c: List[Number],
        constraints: List[Constraint],
        objective: ObjectiveSense = "min",
        bounds: Optional[List[BoundKind]] = None,
        var_names: Optional[List[str]] = None,
        pivot_rule: Rule = "bland",
        verbose: bool = True,
    ):
        self.c_original = [F(v) for v in c]
        self.constraints = constraints
        self.objective = objective
        self.bounds = bounds or [">=0"] * len(c)
        self.original_names = var_names or [f"x{i + 1}" for i in range(len(c))]
        self.pivot_rule = pivot_rule
        self.verbose = verbose

        if len(self.bounds) != len(c):
            raise ValueError("Số lượng điều kiện dấu phải bằng số lượng biến.")

        if len(self.original_names) != len(c):
            raise ValueError("Số lượng tên biến phải bằng số lượng biến.")

        self.order: Dict[str, int] = {}
        self.rows: Dict[str, Row] = {}
        self.basis: List[str] = []
        self.nonbasis: List[str] = []
        self.obj = Row(Fraction(0), {})
        self.phase = ""
        self.canon: Optional[CanonicalLP] = None
        self.slack_names: List[str] = []
        self.x0_name: Optional[str] = None
        self.iteration = 0

    def solve(self):
        self.canon = self._canonicalize_to_min_le_nonnegative()

        need_phase1 = self._build_initial_dictionary_with_x0_phase1(self.canon)

        if self.verbose:
            print("\n=== DẠNG CHUẨN ===")
            print("Hàm mục tiêu đã được chuyển về dạng: min z")
            print("Tất cả ràng buộc đã được chuyển về dạng <=.")
            print("Tất cả biến đã được chuyển về dạng không âm.")
            self._print_canonical_lp()

        if need_phase1:
            phase1_status = self._run_phase1_with_x0()

            if phase1_status == "infeasible":
                self._print_final_conclusion("infeasible", None, None)
                return self._result("infeasible", None, None)
        else:
            if self.verbose:
                print("\nKhông có RHS âm.")
                print("Không cần Pha I.")

        self.phase = "Pha II"
        self._set_objective(self.canon.c_min)

        if self.verbose:
            self._print_dictionary("Từ vựng ban đầu của Pha II")

        status = self._simplex_minimize()

        if status == "unbounded":
            self._print_final_conclusion("unbounded", None, None)
            return self._result("unbounded", None, None)

        internal_point = self._current_internal_point()
        original_point = self._recover_original_solution(internal_point)
        original_value = self._objective_value_original(original_point)

        self._print_final_conclusion("optimal", original_point, original_value)
        return self._result("optimal", original_point, original_value)

    def _new_ordered_var(self, name: str, var_order: List[str]) -> str:
        if name in self.order:
            raise ValueError(f"Tên biến bị trùng: {name}")

        self.order[name] = len(self.order)
        var_order.append(name)
        return name

    def _register_var_if_needed(self, name: str):
        if name not in self.order:
            self.order[name] = len(self.order)

    def _canonicalize_to_min_le_nonnegative(self) -> CanonicalLP:
        c_min_original = self.c_original[:]

        if self.objective == "max":
            c_min_original = [-v for v in c_min_original]

        var_order: List[str] = []
        original_expr: Dict[str, List[Tuple[Fraction, str]]] = {}
        c_min: Dict[str, Fraction] = {}

        for name, bound, cj in zip(self.original_names, self.bounds, c_min_original):
            if bound == ">=0":
                y = self._new_ordered_var(name, var_order)
                original_expr[name] = [(Fraction(1), y)]
                c_min[y] = c_min.get(y, Fraction(0)) + cj

            elif bound == "<=0":
                y = self._new_ordered_var(f"{name}_neg", var_order)
                original_expr[name] = [(Fraction(-1), y)]
                c_min[y] = c_min.get(y, Fraction(0)) - cj

            elif bound == "free":
                yp = self._new_ordered_var(f"{name}_plus", var_order)
                ym = self._new_ordered_var(f"{name}_minus", var_order)

                original_expr[name] = [
                    (Fraction(1), yp),
                    (Fraction(-1), ym),
                ]

                c_min[yp] = c_min.get(yp, Fraction(0)) + cj
                c_min[ym] = c_min.get(ym, Fraction(0)) - cj

            else:
                raise ValueError(f"Điều kiện dấu không được hỗ trợ: {bound}")

        A_le: List[Dict[str, Fraction]] = []
        b_le: List[Fraction] = []

        def transform_coeffs(coeffs_original: List[Number]) -> Dict[str, Fraction]:
            if len(coeffs_original) != len(self.original_names):
                raise ValueError(
                    "Số hệ số trong ràng buộc không khớp với số biến."
                )

            row: Dict[str, Fraction] = {}

            for ai_raw, xname in zip(coeffs_original, self.original_names):
                ai = F(ai_raw)

                for sign, yname in original_expr[xname]:
                    row[yname] = row.get(yname, Fraction(0)) + ai * sign

            return clean_dict(row)

        for cons in self.constraints:
            row = transform_coeffs(cons.coeffs)
            rhs = F(cons.rhs)

            if cons.sense == "<=":
                A_le.append(row)
                b_le.append(rhs)

            elif cons.sense == ">=":
                A_le.append({k: -v for k, v in row.items()})
                b_le.append(-rhs)

            elif cons.sense == "=":
                A_le.append(row)
                b_le.append(rhs)

                A_le.append({k: -v for k, v in row.items()})
                b_le.append(-rhs)

            else:
                raise ValueError(f"Dấu ràng buộc không được hỗ trợ: {cons.sense}")

        return CanonicalLP(
            c_min=clean_dict(c_min),
            A_le=A_le,
            b_le=b_le,
            var_order=var_order,
            original_names=self.original_names,
            original_expr=original_expr,
            original_sense=self.objective,
        )

    def _build_initial_dictionary_with_x0_phase1(self, lp: CanonicalLP) -> bool:
        """
        Xây từ vựng cho:

            min c^T x
            A x <= b
            x >= 0

        Nếu không có RHS âm:

            w_i = b_i - A_i x

        Nếu có RHS âm, tạo từ vựng phụ Pha I:

            w_i = b_i - A_i x + x0

        Pivot đầu tiên bắt buộc:

            biến vào = x0
            biến ra = hàng có RHS âm nhất
        """

        self.rows = {}
        self.basis = []
        self.nonbasis = lp.var_order[:]
        self.slack_names = []
        self.x0_name = None

        need_phase1 = any(b < 0 for b in lp.b_le)

        if need_phase1:
            self.x0_name = "x0"
            self._register_var_if_needed("x0")
            self.nonbasis.append("x0")

        for i, (row_a, b) in enumerate(zip(lp.A_le, lp.b_le), start=1):
            w = f"w{i}"
            self._register_var_if_needed(w)
            self.slack_names.append(w)

            coeffs = {v: -coef for v, coef in row_a.items()}

            if need_phase1:
                coeffs["x0"] = Fraction(1)

            self.rows[w] = Row(b, clean_dict(coeffs))
            self.basis.append(w)

        self.nonbasis = [v for v in self.nonbasis if v not in self.basis]
        self.nonbasis.sort(key=lambda x: self.order[x])

        return need_phase1

    def _choose_phase1_first_leaving(self) -> str:
        """
        Pivot đầu tiên của Pha I:

            biến vào = x0
            biến ra = hàng có RHS âm nhất
        """

        return min(
            self.basis,
            key=lambda basic: (self.rows[basic].const, self.order[basic]),
        )

    def _run_phase1_with_x0(self) -> str:
        self.phase = "Pha I"

        # Mục tiêu Pha I:
        #     min z = x0
        self.obj = Row(Fraction(0), {"x0": Fraction(1)})

        if self.verbose:
            self._print_dictionary("Từ vựng ban đầu của Pha I với x0")

        entering = "x0"
        leaving = self._choose_phase1_first_leaving()

        if self.verbose:
            print("\n[Pha I] Pivot bắt buộc đầu tiên")
            print(f"Biến vào: {entering}")
            print(f"Biến ra:  {leaving}")
            print(
                "Lý do: chọn hàng có RHS âm nhất "
                f"= {fmt(self.rows[leaving].const)}"
            )

        self._pivot(entering, leaving)

        # Sau pivot bắt buộc, x0 có thể trở thành biến cơ sở.
        # Viết lại z = x0 theo các biến không cơ sở hiện tại.
        self._set_objective({"x0": Fraction(1)})

        if self.verbose:
            self._print_dictionary("Từ vựng Pha I sau pivot bắt buộc với x0")

        status = self._simplex_minimize()

        if status == "unbounded":
            raise RuntimeError("Pha I không nên rơi vào trường hợp không giới nội.")

        if self.obj.const != 0:
            return "infeasible"

        self._remove_x0_after_phase1()

        return "feasible"

    def _remove_x0_after_phase1(self):
        x0 = "x0"

        if x0 in self.basis:
            row = self.rows[x0]

            candidates = [
                v
                for v, coef in row.coeffs.items()
                if v != x0 and coef != 0
            ]

            if candidates:
                entering = min(candidates, key=lambda v: self.order[v])

                if self.verbose:
                    print("\nLoại x0 khỏi cơ sở trước khi sang Pha II")
                    print(f"Biến vào: {entering}")
                    print(f"Biến ra:  x0")

                self._pivot(entering, x0)

            else:
                if row.const != 0:
                    raise RuntimeError("x0 có giá trị khác 0 sau Pha I.")

                del self.rows[x0]
                self.basis.remove(x0)

        if x0 in self.nonbasis:
            self.nonbasis.remove(x0)

        for row in self.rows.values():
            row.coeffs.pop(x0, None)

        self.obj.coeffs.pop(x0, None)

    def _set_objective(self, costs: Dict[str, Fraction]):
        z_const = Fraction(0)
        z_coeffs = {v: costs.get(v, Fraction(0)) for v in self.nonbasis}

        for basic in self.basis:
            cb = costs.get(basic, Fraction(0))

            if cb == 0:
                continue

            row = self.rows[basic]
            z_const += cb * row.const

            for nb, coef in row.coeffs.items():
                z_coeffs[nb] = z_coeffs.get(nb, Fraction(0)) + cb * coef

        self.obj = Row(z_const, clean_dict(z_coeffs))

    def _simplex_minimize(self) -> str:
        self.iteration = 0

        while True:
            entering = self._choose_entering()

            if entering is None:
                if self.verbose:
                    print(f"\n[{self.phase}] Đã đạt từ vựng tối ưu.")
                    self._print_current_point()

                return "optimal"

            leaving = self._choose_leaving(entering)

            if leaving is None:
                if self.verbose:
                    print(f"\n[{self.phase}] Không giới nội theo biến {entering}.")

                return "unbounded"

            self.iteration += 1

            if self.verbose:
                print(f"\n[{self.phase}] Lần lặp {self.iteration}")
                print(f"Biến vào: {entering}")
                print(f"Biến ra:  {leaving}")

            self._pivot(entering, leaving)

            if self.verbose:
                self._print_dictionary(f"Từ vựng sau pivot {self.iteration}")

    def _choose_entering(self) -> Optional[str]:
        """
        Với từ vựng bài toán min:

            z = z0 + sum c_j x_j

        Nếu có c_j < 0 thì tăng x_j có thể làm z giảm.
        """

        candidates = [
            v for v in self.nonbasis if self.obj.coeffs.get(v, Fraction(0)) < 0
        ]

        if not candidates:
            return None

        if self.pivot_rule == "bland":
            return min(candidates, key=lambda x: self.order[x])

        if self.pivot_rule == "dantzig":
            return min(
                candidates,
                key=lambda x: (self.obj.coeffs.get(x, Fraction(0)), self.order[x]),
            )

        raise ValueError(f"Quy tắc chọn pivot không được hỗ trợ: {self.pivot_rule}")

    def _choose_leaving(self, entering: str) -> Optional[str]:
        ratios: List[Tuple[Fraction, int, str]] = []

        for basic in self.basis:
            row = self.rows[basic]
            a = row.coeffs.get(entering, Fraction(0))

            if a < 0:
                ratios.append((row.const / (-a), self.order[basic], basic))

        if not ratios:
            return None

        ratios.sort()
        return ratios[0][2]

    def _pivot(self, entering: str, leaving: str):
        old_row = self.rows[leaving]
        a_e = old_row.coeffs[entering]

        if a_e == 0:
            raise ZeroDivisionError("Pivot không hợp lệ: hệ số pivot bằng 0.")

        new_coeffs: Dict[str, Fraction] = {}
        new_const = -old_row.const / a_e

        new_coeffs[leaving] = Fraction(1) / a_e

        for nb, coef in old_row.coeffs.items():
            if nb == entering:
                continue

            new_coeffs[nb] = -coef / a_e

        new_row = Row(new_const, clean_dict(new_coeffs))

        def substitute(expr: Row) -> Row:
            c = expr.coeffs.get(entering, Fraction(0))

            if c == 0:
                return expr

            out_const = expr.const + c * new_row.const
            out_coeffs = dict(expr.coeffs)

            del out_coeffs[entering]

            for v, coef in new_row.coeffs.items():
                out_coeffs[v] = out_coeffs.get(v, Fraction(0)) + c * coef

            return Row(out_const, clean_dict(out_coeffs))

        for basic in list(self.basis):
            if basic == leaving:
                continue

            self.rows[basic] = substitute(self.rows[basic])

        self.obj = substitute(self.obj)

        del self.rows[leaving]
        self.rows[entering] = new_row

        self.basis = [entering if b == leaving else b for b in self.basis]
        self.nonbasis = [leaving if v == entering else v for v in self.nonbasis]
        self.nonbasis.sort(key=lambda x: self.order[x])

    def _print_canonical_lp(self):
        assert self.canon is not None

        print("min z =", self._linear_to_str(self.canon.c_min, const=Fraction(0)))

        for i, (row, b) in enumerate(zip(self.canon.A_le, self.canon.b_le), start=1):
            print(f"Ràng buộc {i}: {self._linear_to_str(row, const=Fraction(0))} <= {fmt(b)}")

        print("Ánh xạ biến gốc:")

        for x in self.original_names:
            parts = []

            for coef, y in self.canon.original_expr[x]:
                parts.append((coef, y))

            print(
                f"  {x} = "
                f"{self._linear_to_str(dict((y, coef) for coef, y in parts), const=Fraction(0))}"
            )

    def _print_dictionary(self, title: str):
        print(f"\n--- {title} ---")
        print("z =", self._row_expr_to_str(self.obj))
        print("--------")

        for basic in self.basis:
            print(f"{basic} = {self._row_expr_to_str(self.rows[basic])}")

        self._print_current_point()

    def _print_current_point(self):
        internal = self._current_internal_point()
        original = self._recover_original_solution(internal)

        obj_val = self._objective_value_original(original)

        print("Điểm của từ vựng hiện tại khi cho tất cả biến không cơ sở bằng 0:")

        print(
            "  biến nội bộ:",
            ", ".join(
                f"{v}={fmt(internal.get(v, Fraction(0)))}"
                for v in sorted(internal, key=lambda x: self.order.get(x, 10**9))
            ),
        )

        print(
            "  biến gốc:",
            ", ".join(f"{x}={fmt(original[x])}" for x in self.original_names),
        )

        print(f"  giá trị hàm mục tiêu gốc = {fmt(obj_val)}")

    def _linear_to_str(
        self,
        coeffs: Dict[str, Fraction],
        const: Fraction = Fraction(0),
    ) -> str:
        return self._expr_to_str(const, coeffs)

    def _row_expr_to_str(self, row: Row) -> str:
        return self._expr_to_str(row.const, row.coeffs)

    def _expr_to_str(self, const: Fraction, coeffs: Dict[str, Fraction]) -> str:
        terms: List[str] = []

        if const != 0 or not coeffs:
            terms.append(fmt(const))

        for v in sorted(coeffs, key=lambda x: self.order.get(x, 10**9)):
            coef = coeffs[v]

            if coef == 0:
                continue

            sign = "+" if coef > 0 else "-"
            abscoef = abs(coef)
            atom = v if abscoef == 1 else f"{fmt(abscoef)}{v}"

            if not terms:
                terms.append(atom if coef > 0 else f"-{atom}")
            else:
                terms.append(f"{sign} {atom}")

        return " ".join(terms) if terms else "0"

    def _current_internal_point(self) -> Dict[str, Fraction]:
        point: Dict[str, Fraction] = {}

        for v in self.nonbasis:
            point[v] = Fraction(0)

        for b in self.basis:
            point[b] = self.rows[b].const

        return point

    def _recover_original_solution(
        self,
        internal: Dict[str, Fraction],
    ) -> Dict[str, Fraction]:
        assert self.canon is not None

        ans: Dict[str, Fraction] = {}

        for x in self.original_names:
            val = Fraction(0)

            for coef, y in self.canon.original_expr[x]:
                val += coef * internal.get(y, Fraction(0))

            ans[x] = val

        return ans

    def _objective_value_original(self, original_point: Dict[str, Fraction]) -> Fraction:
        val = Fraction(0)

        for ci, x in zip(self.c_original, self.original_names):
            val += ci * original_point[x]

        return val

    def _print_final_conclusion(self, status: str, point, value):
        if not self.verbose:
            return

        print("\n================ KẾT LUẬN ================")

        if status == "optimal":
            print("Trạng thái: tối ưu")

            if point is not None:
                print(
                    "Nghiệm tối ưu:",
                    ", ".join(f"{x}={fmt(point[x])}" for x in self.original_names),
                )

            if self.objective == "min":
                print(f"min z = {fmt(value)}")
            else:
                print(f"max z = {fmt(value)}")

        elif status == "unbounded":
            print("Trạng thái: không giới nội")

            if self.objective == "min":
                print("min z = -infty")
            else:
                print("max z = +infty")

        elif status == "infeasible":
            print("Trạng thái: vô nghiệm")

            if self.objective == "min":
                print("min z = +infty")
            else:
                print("max z = -infty")

        else:
            print(f"Trạng thái không xác định: {status}")

        print("==========================================")

    def _result(self, status: str, point, value):
        return {
            "status": status,
            "objective_sense": self.objective,
            "optimal_value": value,
            "solution": point,
        }