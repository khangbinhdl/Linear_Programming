import streamlit as st
import os
import sys
import random
import io
import contextlib
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN IMPORT
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Simplex.simplex_dictionary_solver import SimplexDictionarySolver, Constraint, fmt
from Graphical.graphical_lp_2d import plot_lp_2d, Constraint2D, fmt_float

# --- 1. CẤU HÌNH GIAO DIỆN & CSS ---
st.set_page_config(page_title="Quy Hoạch Tuyến Tính", layout="wide")

st.markdown("""
    <style>
    /* Ép text thẳng hàng */
    .inline-text {
        display: flex; align-items: center; justify-content: center;
        height: 2.5rem; font-size: 1.1rem; font-weight: 500; margin-top: 2px;
    }
    /* Thu hẹp khoảng cách giữa các cột */
    div[data-testid="column"] { padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
    
    /* Căn giữa chữ và số trong tất cả các ô nhập liệu */
    .stTextInput > div > div > input { text-align: center; font-weight: bold; }
    .stNumberInput > div > div > input { text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0d6efd; margin-bottom: 30px;'>Phần Mềm Giải Quy Hoạch Tuyến Tính</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📐 Phương pháp Simplex", "📈 Phương pháp Hình học (2 biến)"])

BOUNDS_MAP = {">=0": ">=0", "<=0": "<=0", "Tự do": "free"}

# Hàm bổ trợ: Trả về giá trị số kèm trạng thái xem có chuyển đổi thành công hay không
def parse_float_with_status(val_str):
    try:
        return float(val_str), True
    except ValueError:
        return 0.0, False

# ==========================================
# CÁC HÀM RANDOM (CALLBACKS)
# ==========================================
def randomize_simplex(n_vars, n_cons):
    st.session_state["s_obj"] = random.choice(["max", "min"])
    for i in range(n_vars):
        st.session_state[f"obj_{i}"] = str(random.randint(-10, 10))
        st.session_state[f"bound_{i}"] = random.choice([">=0", "<=0", "Tự do"])
    for i in range(n_cons):
        for j in range(n_vars):
            st.session_state[f"c_{i}_{j}"] = str(random.randint(-10, 10))
        st.session_state[f"rhs_{i}"] = str(random.randint(1, 20))
        st.session_state[f"sense_{i}"] = random.choice(["<=", ">=", "="])

def randomize_graphical(n_cons):
    st.session_state["g_obj"] = random.choice(["max", "min"])
    st.session_state["g_bx1"] = random.choice([">=0", "<=0", "Tự do"])
    st.session_state["g_bx2"] = random.choice([">=0", "<=0", "Tự do"])
    st.session_state["g_c1"] = str(random.randint(-10, 10))
    st.session_state["g_c2"] = str(random.randint(-10, 10))
    for i in range(n_cons):
        st.session_state[f"ga1_{i}"] = str(random.randint(-10, 10))
        st.session_state[f"ga2_{i}"] = str(random.randint(-10, 10))
        st.session_state[f"grhs_{i}"] = str(random.randint(1, 20))
        st.session_state[f"gsense_{i}"] = random.choice(["<=", ">=", "="])


# ==========================================
# TAB 1: SIMPLEX
# ==========================================
with tab1:
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    with col1:
        num_vars = st.number_input("Số lượng biến", min_value=1, value=3, step=1, key="s_vars")
    with col2:
        num_cons = st.number_input("Số lượng ràng buộc", min_value=1, value=3, step=1, key="s_cons")
    with col3:
        pivot_rule = st.selectbox("Quy tắc Pivot", ["bland", "dantzig"], key="s_pivot")
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🎲 Tạo số liệu ngẫu nhiên", on_click=randomize_simplex, args=(num_vars, num_cons), key="btn_rand_s")
    
    st.divider()

    # Cờ theo dõi lỗi nhập liệu của Tab Simplex
    input_error_s = False

    # --- HÀM MỤC TIÊU ---
    st.write("### Hàm mục tiêu (Z)")
    obj_cols = st.columns([1.5, 0.8] + [1.2, 0.8] * num_vars + [3])
    obj_sense = obj_cols[0].selectbox("Mục tiêu", ["max", "min"], key="s_obj", label_visibility="collapsed")
    obj_cols[1].markdown('<div class="inline-text">Z &nbsp;= </div>', unsafe_allow_html=True)
    
    obj_coeffs = []
    for i in range(num_vars):
        val_str = obj_cols[2 + i*2].text_input(f"c{i}", value="1", key=f"obj_{i}", label_visibility="collapsed")
        val, is_ok = parse_float_with_status(val_str)
        if not is_ok:
            input_error_s = True
        obj_coeffs.append(val)
        suffix = f"x<sub>{i+1}</sub> &nbsp;+" if i < num_vars - 1 else f"x<sub>{i+1}</sub>"
        obj_cols[3 + i*2].markdown(f'<div class="inline-text">{suffix}</div>', unsafe_allow_html=True)

    # --- HỆ RÀNG BUỘC ---
    st.write("### Hệ ràng buộc")
    constraints_data = []
    for i in range(num_cons):
        c_cols = st.columns([0.5] + [1.2, 0.8] * num_vars + [1.2, 1.5] + [3])
        coeffs = []
        for j in range(num_vars):
            val_str = c_cols[1 + j*2].text_input(f"x{j+1}", value="1", key=f"c_{i}_{j}", label_visibility="collapsed")
            val, is_ok = parse_float_with_status(val_str)
            if not is_ok:
                input_error_s = True
            coeffs.append(val)
            suffix = f"x<sub>{j+1}</sub> &nbsp;+" if j < num_vars - 1 else f"x<sub>{j+1}</sub>"
            c_cols[2 + j*2].markdown(f'<div class="inline-text">{suffix}</div>', unsafe_allow_html=True)
        
        sense = c_cols[-3].selectbox("Dấu", ["<=", ">=", "="], key=f"sense_{i}", label_visibility="collapsed")
        rhs_str = c_cols[-2].text_input("RHS", value="10", key=f"rhs_{i}", label_visibility="collapsed")
        rhs_val, is_ok = parse_float_with_status(rhs_str)
        if not is_ok:
            input_error_s = True
        constraints_data.append((coeffs, sense, rhs_val))
    
    # --- ĐIỀU KIỆN DẤU ---
    st.write("### Điều kiện dấu (Bounds)")
    bounds = []
    bound_cols = st.columns([1] * num_vars + [5])
    for i in range(num_vars):
        with bound_cols[i]:
            b_display = st.selectbox(
                f"Dấu x{i+1}", [">=0", "<=0", "Tự do"], key=f"bound_{i}",
                format_func=lambda x: f"x{i+1} {x}" if x != "Tự do" else f"x{i+1} Tùy ý"
            )
            bounds.append(BOUNDS_MAP[b_display])
            
    st.write("###")

    # HIỂN THỊ THÔNG BÁO LỖI NẾU CÓ KÝ TỰ LẠ
    if input_error_s:
        st.error("⚠️ Bạn không nhập đúng xin hãy nhập lại")

    if st.button("🚀 Giải bài toán (Simplex)", type="primary", use_container_width=True):
        if input_error_s:
            st.warning("Không thể tính toán! Vui lòng kiểm tra và sửa lại các ô nhập liệu bị lỗi.")
        else:
            cons_list = [Constraint(c[0], c[1], c[2]) for c in constraints_data]
            var_names = [f"x{i+1}" for i in range(num_vars)]
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    solver = SimplexDictionarySolver(
                        c=obj_coeffs, constraints=cons_list, objective=obj_sense,
                        bounds=bounds, var_names=var_names, pivot_rule=pivot_rule, verbose=True
                    )
                    result = solver.solve()
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                    result = None
                    
            output = f.getvalue()
            
            if result:
                st.divider()
                st.subheader("📊 Kết quả")
                if result['status'] == "optimal":
                    st.success(f"**Trạng thái:** TỐI ƯU (Optimal)")
                    st.info(f"**Giá trị tối ưu (Z) =** {fmt(result['optimal_value'])}")
                    if result.get("solution"):
                        sol_str = "  |  ".join([f"**{k}** = {fmt(v)}" for k, v in result["solution"].items()])
                        st.success(f"**Nghiệm:** {sol_str}")
                elif result['status'] == "infeasible":
                    st.error(f"**Trạng thái:** VÔ NGHIỆM (Infeasible)")
                elif result['status'] == "unbounded":
                    st.warning(f"**Trạng thái:** KHÔNG GIỚI NỘI (Unbounded)")

            with st.expander("📝 Xem chi tiết các bước giải thuật", expanded=False):
                st.code(output, language="text")


# ==========================================
# TAB 2: GRAPHICAL
# ==========================================
with tab2:
    st.info("💡 Phương pháp hình học chỉ áp dụng cho bài toán đúng 2 biến (x1, x2).")
    
    col_g1, col_g2, col_g3 = st.columns([1, 1, 2])
    with col_g1:
        num_cons_g = st.number_input("Số lượng ràng buộc", min_value=1, value=3, step=1, key="g_ncons")
    with col_g3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🎲 Tạo số liệu ngẫu nhiên", on_click=randomize_graphical, args=(int(num_cons_g),), key="btn_rand_g")
    
    st.divider()

    # Cờ theo dõi lỗi nhập liệu của Tab Hình Học
    input_error_g = False

    # --- HÀM MỤC TIÊU ---
    st.write("### Hàm mục tiêu (Z)")
    obj_cols_g = st.columns([1.5, 0.8, 1.2, 0.8, 1.2, 0.8, 4])
    obj_sense_g = obj_cols_g[0].selectbox("Mục tiêu", ["max", "min"], key="g_obj", label_visibility="collapsed")
    obj_cols_g[1].markdown('<div class="inline-text">Z &nbsp;= </div>', unsafe_allow_html=True)
    
    c1_str = obj_cols_g[2].text_input("c1", value="1", key="g_c1", label_visibility="collapsed")
    c1, is_ok = parse_float_with_status(c1_str)
    if not is_ok:
        input_error_g = True

    obj_cols_g[3].markdown('<div class="inline-text">x<sub>1</sub> &nbsp;+ </div>', unsafe_allow_html=True)
    
    c2_str = obj_cols_g[4].text_input("c2", value="1", key="g_c2", label_visibility="collapsed")
    c2, is_ok = parse_float_with_status(c2_str)
    if not is_ok:
        input_error_g = True

    obj_cols_g[5].markdown('<div class="inline-text">x<sub>2</sub></div>', unsafe_allow_html=True)
        
    # --- HỆ RÀNG BUỘC ---
    st.write("### Hệ ràng buộc")
    constraints_g = []
    for i in range(int(num_cons_g)):
        cg_cols = st.columns([0.5, 1.2, 0.8, 1.2, 0.8, 1.2, 1.5, 3])
        
        a1_str = cg_cols[1].text_input("a1", value="1", key=f"ga1_{i}", label_visibility="collapsed")
        a1, is_ok = parse_float_with_status(a1_str)
        if not is_ok:
            input_error_g = True

        cg_cols[2].markdown('<div class="inline-text">x<sub>1</sub> &nbsp;+ </div>', unsafe_allow_html=True)
        
        a2_str = cg_cols[3].text_input("a2", value="1", key=f"ga2_{i}", label_visibility="collapsed")
        a2, is_ok = parse_float_with_status(a2_str)
        if not is_ok:
            input_error_g = True

        cg_cols[4].markdown('<div class="inline-text">x<sub>2</sub></div>', unsafe_allow_html=True)
        sense = cg_cols[5].selectbox("Dấu", ["<=", ">=", "="], key=f"gsense_{i}", label_visibility="collapsed")
        
        rhs_str = cg_cols[6].text_input("RHS", value="10", key=f"grhs_{i}", label_visibility="collapsed")
        rhs, is_ok = parse_float_with_status(rhs_str)
        if not is_ok:
            input_error_g = True

        constraints_g.append((a1, a2, sense, rhs))
        
    # --- ĐIỀU KIỆN DẤU ---
    st.write("### Điều kiện dấu (Bounds)")
    bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 5])
    bx1 = BOUNDS_MAP[bcol1.selectbox("Dấu x1", [">=0", "<=0", "Tự do"], key="g_bx1", format_func=lambda x: f"x1 {x}" if x != "Tự do" else "x1 Tùy ý")]
    bx2 = BOUNDS_MAP[bcol2.selectbox("Dấu x2", [">=0", "<=0", "Tự do"], key="g_bx2", format_func=lambda x: f"x2 {x}" if x != "Tự do" else "x2 Tùy ý")]
        
    st.write("###")

    # HIỂN THỊ THÔNG BÁO LỖI NẾU CÓ KÝ TỰ LẠ
    if input_error_g:
        st.error("⚠️ Bạn không nhập đúng xin hãy nhập lại")

    if st.button("📈 Vẽ Đồ Thị (Hình học)", type="primary", use_container_width=True, key="g_solve"):
        if input_error_g:
            st.warning("Không thể vẽ đồ thị! Vui lòng kiểm tra và sửa lại các ô nhập liệu bị lỗi.")
        else:
            cons_list = [Constraint2D(c[0], c[1], c[2], c[3], "") for c in constraints_g]
            
            with st.spinner("Đang vẽ đồ thị..."):
                try:
                    result = plot_lp_2d(
                        c=(c1, c2), objective=obj_sense_g, constraints=cons_list,
                        bounds=(bx1, bx2), show=False
                    )
                    
                    st.divider()
                    st.subheader("📊 Kết quả Đồ họa")
                    
                    optimal_type = result.get("optimal_type", "")
                    status_msg = result.get("status", "")
                    
                    if optimal_type in ["single", "multiple"]:
                        st.success(f"**Trạng thái:** {status_msg}")
                        st.info(f"**Giá trị tối ưu (Z) =** {fmt_float(result['optimal_value'])}")
                    elif optimal_type == "none":
                        st.error(f"**Trạng thái:** {status_msg}")
                    else:
                        st.info(f"**Trạng thái:** {status_msg}")
                    
                    if "figure" in result:
                        st.pyplot(result["figure"])
                        plt.close(result["figure"]) 
                        
                except Exception as e:
                    st.error(f"Lỗi trong quá trình giải: {str(e)}")