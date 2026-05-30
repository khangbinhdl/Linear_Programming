# Linear Programming Solver

Dự án này gồm hai phần chính:

1. `Simplex/simplex_dictionary_solver.py`  
   Giải bài toán quy hoạch tuyến tính bằng **phương pháp đơn hình dạng từ vựng**.  
   Hỗ trợ:
    - Dantzig's Simplex Algorithm
    - Bland's Rule
    - Two-Phase Simplex Method

2. `Graphical/graphical_lp_2d.py`  
   Giải và minh họa bài toán quy hoạch tuyến tính **2 biến** bằng **phương pháp hình học**, vẽ miền chấp nhận được, đường mục tiêu, vector pháp tuyến và nghiệm tối ưu.

---

## 1. Cấu trúc thư mục

```text
.
├── Graphical/
│   └── graphical_lp_2d.py
│
├── Simplex/
│   └── simplex_dictionary_solver.py
│
├── README.md
└── requirements.txt
```

## 2. Cài đặt
Cài thư viện
```bash
pip install -r requirements.txt
```


## 3. Simplex Dictionary Solver

**File chính**: `Simplex/simplex_dictionary_solver.py`

### 3.1. Chức năng chính

Solver hỗ trợ:

- Bài toán `min` và `max`.
- Tự chuyển `max` về `min`.
- **Ràng buộc:**
  - `<=`
  - `>=`
  - `=`
- **Biến:**
  - `>=0`
  - `<=0`
  - `free`
- **Tự tách biến tự do:** `x = x_plus - x_minus`
- **Biến bù được đặt tên:** `w1, w2, w3, ...`
- **Pha I dùng một biến phụ duy nhất:** `x0`
  - Từ vựng Pha I có dạng: `w_i = b_i - A_i x + x0`
  - Pivot đầu tiên của Pha I: biến vào `x0`, biến ra là hàng có RHS âm nhất.
- **Có hai quy tắc chọn biến vào:**
  - `bland`
  - `dantzig`
- **In từng bước:**
  - Pha hiện tại
  - Từ vựng
  - Biến vào, biến ra
  - Điểm hiện tại của từ vựng
  - Nghiệm theo biến gốc
  - Kết luận tối ưu, vô nghiệm hoặc không giới nội

### 3.2. Ví dụ sử dụng Simplex

**Bài toán:**
```text
max z = 2x1 - 6x2

s.t.
    -x1 - x2 - x3 <= -2
     2x1 - x2 + x3 <= 1

    x1, x2, x3 >= 0
```

**Code:**
```python
from Simplex.simplex_dictionary_solver import (
    Constraint,
    SimplexDictionarySolver,
    fmt,
)

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

result = solver.solve()

print("Trạng thái:", result["status"])

if result["solution"] is not None:
    print("Nghiệm:", {k: fmt(v) for k, v in result["solution"].items()})
    print("Giá trị tối ưu:", fmt(result["optimal_value"]))
```

### 3.3. Ví dụ có biến tự do và ràng buộc dấu bằng

**Bài toán:**
```text
min z = x1 - x2

s.t.
    x1 + x2 = 4
    x1 - x2 >= 2
    x1 <= 5

    x1 >= 0
    x2 free
```

**Code:**
```python
from Simplex.simplex_dictionary_solver import (
    Constraint,
    SimplexDictionarySolver,
    fmt,
)

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

result = solver.solve()

print("Trạng thái:", result["status"])

if result["solution"] is not None:
    print("Nghiệm:", {k: fmt(v) for k, v in result["solution"].items()})
    print("Giá trị tối ưu:", fmt(result["optimal_value"]))
```

### 3.4. Chọn Bland rule hoặc Dantzig rule

Dùng Bland:
```python
pivot_rule="bland"
```

Dùng Dantzig:
```python
pivot_rule="dantzig"
```

Ví dụ:
```python
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
```

### 3.5. Kết quả trả về

Hàm `solve()` trả về dictionary:
```python
{
    "status": "optimal" | "infeasible" | "unbounded",
    "objective_sense": "min" | "max",
    "optimal_value": value_or_None,
    "solution": solution_dict_or_None,
}
```

Ví dụ kết quả tối ưu:
```python
{
    "status": "optimal",
    "objective_sense": "max",
    "optimal_value": Fraction(12, 1),
    "solution": {
        "x1": Fraction(4, 1),
        "x2": Fraction(0, 1),
    },
}
```

### 3.6. Ý nghĩa trạng thái

- **Nếu tối ưu:** Trạng thái: `tối ưu`
- **Nếu không giới nội:** Trạng thái: `không giới nội`
  - Với bài toán `min`: `min z = -infty`
  - Với bài toán `max`: `max z = +infty`
- **Nếu vô nghiệm:** Trạng thái: `vô nghiệm`
  - Với bài toán `min`: `min z = +infty`
  - Với bài toán `max`: `max z = -infty`

---

## 4. Graphical LP 2D

**File chính**: `Graphical/graphical_lp_2d.py`

### 4.1. Chức năng chính

File này dùng để giải và vẽ bài toán quy hoạch tuyến tính 2 biến. Hỗ trợ:

- Bài toán `min` và `max`.
- **Ràng buộc:** `<=`, `>=`, `=`
- **Điều kiện dấu:** `x >= 0`, `x <= 0`, `x free`
- Vẽ các đường ràng buộc.
- Vẽ ràng buộc dấu.
- Vẽ miền chấp nhận được.
- Đánh dấu các đỉnh của miền chấp nhận được.
- Vẽ đường mục tiêu mẫu: `z = |lcm(c1, c2)|`
- Vẽ vector pháp tuyến của đường mục tiêu.
- Vẽ vector pháp tuyến của từng ràng buộc, chỉ về phía miền chấp nhận được.
- Vẽ đường mục tiêu tối ưu.
- **Phát hiện trường hợp:**
  - Nghiệm tối ưu duy nhất
  - Vô số nghiệm tối ưu trên một đoạn thẳng

### 4.2. Ví dụ sử dụng Graphical Solver

**Bài toán:**
```text
max z = 2x + 3y

s.t.
    x + y <= 4
    x <= 2
    y <= 3

    x >= 0
    y >= 0
```

**Code:**
```python
from Graphical.graphical_lp_2d import Constraint2D, plot_lp_2d

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
print("Các đỉnh:", result["vertices"])
print("Kiểu nghiệm tối ưu:", result["optimal_type"])
print("Điểm tối ưu:", result.get("optimal_point"))
print("Giá trị tối ưu:", result["optimal_value"])
```

### 4.3. Ví dụ có vô số nghiệm tối ưu

**Bài toán:**
```text
min z = x + 2y

s.t.
    3x + y >= 3
    x + 2y >= 4
    x - y <= 1
    x <= 5
    y <= 5

    x >= 0
    y >= 0
```

**Code:**
```python
from Graphical.graphical_lp_2d import Constraint2D, plot_lp_2d

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
    title="Bài toán có vô số nghiệm tối ưu",
    save_path="multiple_optima.png",
    show=True,
)

print("Trạng thái:", result["status"])
print("Kiểu nghiệm:", result["optimal_type"])
print("Các đỉnh tối ưu:", result["optimal_vertices"])
print("Giá trị tối ưu:", result["optimal_value"])
```

### 4.4. Kết quả trả về của plot_lp_2d

Hàm `plot_lp_2d()` trả về dictionary.

**Trường hợp nghiệm tối ưu duy nhất:**
```python
{
    "status": "có nghiệm tối ưu duy nhất trên các đỉnh hữu hạn tìm được",
    "vertices": [...],
    "optimal_type": "single",
    "optimal_vertices": [...],
    "optimal_point": (x_star, y_star),
    "optimal_value": z_star,
    "figure": fig,
    "axis": ax,
    "legend_axis": legend_ax,
}
```

**Trường hợp vô số nghiệm tối ưu trên một đoạn thẳng:**
```python
{
    "status": "có vô số nghiệm tối ưu trên một đoạn thẳng",
    "vertices": [...],
    "optimal_type": "multiple",
    "optimal_vertices": [...],
    "optimal_segment": [...],
    "optimal_value": z_star,
    "figure": fig,
    "axis": ax,
    "legend_axis": legend_ax,
}
```

**Trường hợp không tìm được đỉnh hữu hạn:**
```python
{
    "status": "không tìm được đỉnh khả thi hữu hạn",
    "vertices": [...],
    "optimal_type": "none",
    "optimal_vertices": [],
    "optimal_value": None,
    "figure": fig,
    "axis": ax,
    "legend_axis": legend_ax,
}
```