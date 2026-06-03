# Linear Programming Solver

Phần mềm giải bài toán quy hoạch tuyến tính với giao diện web bằng Streamlit, gồm hai hướng xử lý chính:

1. `Simplex/simplex_dictionary_solver.py`: giải bài toán bằng phương pháp đơn hình dạng từ vựng.
2. `Graphical/graphical_lp_2d.py`: giải bài toán 2 biến bằng phương pháp hình học và hiển thị bằng Plotly.

## Cấu trúc dự án

```text
.
├── frontend/
│   └── app.py
├── Graphical/
│   └── graphical_lp_2d.py
├── Simplex/
│   └── simplex_dictionary_solver.py
├── Makefile
├── README.md
└── requirements.txt
```

## Cài đặt

### Cách nhanh bằng Make

```bash
make make-install
```

### Cách thủ công

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

### Cách nhanh bằng Make

```bash
make make-run
```

Chạy với port tùy chọn:

```bash
make make-run PORT=8502
```

### Cách thủ công

```bash
streamlit run frontend/app.py --server.port 8501
```

Nếu muốn đổi port, thay `8501` bằng số port bạn cần.

Sau khi chạy, ứng dụng sẽ mở tại `http://localhost:8501`.

## Makefile

Makefile hiện có 2 target:

- `make-install`: cài toàn bộ thư viện từ `requirements.txt`.
- `make-run`: chạy giao diện Streamlit ở `frontend/app.py`.
- Có thể truyền `PORT=8502` hoặc giá trị khác khi chạy `make make-run`.

## Simplex Solver

File chính: `Simplex/simplex_dictionary_solver.py`

### Hỗ trợ

- Bài toán `min` và `max`.
- Ràng buộc `<=`, `>=`, `=`.
- Điều kiện dấu biến `>=0`, `<=0`, `free`.
- Tách biến tự do tự động.
- Hai quy tắc chọn biến vào: `bland` và `dantzig`.
- Trả về trạng thái `optimal`, `infeasible`, hoặc `unbounded`.

### Ví dụ

```python
from Simplex.simplex_dictionary_solver import Constraint, SimplexDictionarySolver, fmt

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

result = solver.solve()

print("Trạng thái:", result["status"])
print("Giá trị tối ưu:", fmt(result["optimal_value"]))
print("Nghiệm:", {k: fmt(v) for k, v in result["solution"].items()})
```

## Graphical LP 2D

File chính: `Graphical/graphical_lp_2d.py`

### Hỗ trợ

- Bài toán `min` và `max` với đúng 2 biến.
- Ràng buộc `<=`, `>=`, `=`.
- Điều kiện dấu `x >= 0`, `x <= 0`, `x free`.
- Vẽ miền khả thi, đỉnh, đường mục tiêu mẫu và đường mục tiêu tối ưu.
- Vẽ vector pháp tuyến của ràng buộc và của hàm mục tiêu.
- Phát hiện nghiệm tối ưu duy nhất hoặc vô số nghiệm tối ưu trên một đoạn thẳng.
- Hiển thị bằng Plotly để dùng tương tác tốt trong Streamlit.

### Ví dụ

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
    title="Ví dụ: max z = 2x + 3y",
    show=True,
)

print("Trạng thái:", result["status"])
print("Kiểu nghiệm tối ưu:", result["optimal_type"])
print("Giá trị tối ưu:", result["optimal_value"])
```

## Ghi chú

- Giao diện chính nằm ở `frontend/app.py`.
- Nếu bạn thay đổi logic đồ họa trong `Graphical/graphical_lp_2d.py`, hãy nhớ cập nhật README để đồng bộ với code thực tế.
