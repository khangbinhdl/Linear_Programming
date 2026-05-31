import streamlit as st
import os
import sys
import random

# Thêm đường dẫn để import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Simplex.simplex_dictionary_solver import SimplexDictionarySolver, Constraint, fmt
from Graphical.graphical_lp_2d import plot_lp_2d, Constraint2D, fmt_float
import io
import contextlib

st.set_page_config(page_title="Quy Hoạch Tuyến Tính", layout="wide")
st.title("Phần Mềm Giải Quy Hoạch Tuyến Tính (Streamlit)")

tab1, tab2 = st.tabs(["Phương pháp Simplex", "Phương pháp Hình học (2 biến)"])

# Từ điển ánh xạ hiển thị dấu
BOUNDS_MAP = {">=0": ">=0", "<=0": "<=0", "Tự do": "free"}

with tab1:
    st.header("Phương pháp Simplex")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_vars = st.number_input("Số lượng biến", min_value=1, value=2, step=1, key="s_vars")
    with col2:
        num_cons = st.number_input("Số lượng ràng buộc", min_value=1, value=2, step=1, key="s_cons")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Tạo ngẫu nhiên (Simplex)"):
            st.session_state["s_obj"] = random.choice(["max", "min"])
            for i in range(num_vars):
                st.session_state[f"obj_{i}"] = float(random.randint(-10, 10))
                st.session_state[f"bound_{i}"] = random.choice([">=0", "<=0", "Tự do"])
            for i in range(num_cons):
                for j in range(num_vars):
                    st.session_state[f"c_{i}_{j}"] = float(random.randint(-10, 10))
                st.session_state[f"rhs_{i}"] = float(random.randint(-20, 20))
                st.session_state[f"sense_{i}"] = random.choice(["<=", ">=", "="])

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        obj_sense = st.selectbox("Mục tiêu", ["max", "min"], key="s_obj")
    with col_opt2:
        pivot_rule = st.selectbox("Quy tắc chọn biến (Pivot Rule)", ["bland", "dantzig"], key="s_pivot")
    
    st.subheader("Hàm mục tiêu (Nhập hệ số)")
    obj_cols = st.columns(num_vars)
    obj_coeffs = []
    for i in range(num_vars):
        with obj_cols[i]:
            val = st.number_input(f"Hệ số x{i+1}", value=1.0, key=f"obj_{i}")
            obj_coeffs.append(val)
            
    st.subheader("Ràng buộc dấu (Dành cho các biến)")
    bounds = []
    bound_cols = st.columns(num_vars)
    for i in range(num_vars):
        with bound_cols[i]:
            b_display = st.selectbox(f"Dấu x{i+1}", [">=0", "<=0", "Tự do"], key=f"bound_{i}")
            bounds.append(BOUNDS_MAP[b_display])
            
    st.subheader("Các ràng buộc hệ thống")
    constraints_data = []
    for i in range(num_cons):
        st.markdown(f"**Ràng buộc {i+1}**")
        c_cols = st.columns(num_vars + 2)
        coeffs = []
        for j in range(num_vars):
            with c_cols[j]:
                coeffs.append(st.number_input(f"x{j+1}", value=1.0, key=f"c_{i}_{j}"))
        with c_cols[num_vars]:
            sense = st.selectbox("Dấu", ["<=", ">=", "="], key=f"sense_{i}")
        with c_cols[num_vars+1]:
            rhs = st.number_input("RHS", value=10.0, key=f"rhs_{i}")
        constraints_data.append((coeffs, sense, rhs))
        
    if st.button("Giải bằng Simplex", type="primary"):
        cons_list = [Constraint(c[0], c[1], c[2]) for c in constraints_data]
        var_names = [f"x{i+1}" for i in range(num_vars)]
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                solver = SimplexDictionarySolver(
                    c=obj_coeffs,
                    constraints=cons_list,
                    objective=obj_sense,
                    bounds=bounds,
                    var_names=var_names,
                    pivot_rule=pivot_rule,
                    verbose=True
                )
                result = solver.solve()
            except Exception as e:
                print(f"Lỗi: {str(e)}")
                result = None
                
        output = f.getvalue()
        
        if result:
            if result['status'] == "optimal":
                st.success(f"Trạng thái: Tối ưu (Optimal)")
            elif result['status'] == "infeasible":
                st.error(f"Trạng thái: Vô nghiệm (Infeasible)")
            elif result['status'] == "unbounded":
                st.warning(f"Trạng thái: Không giới nội (Unbounded)")
            else:
                st.info(f"Trạng thái: {result['status']}")

            if result.get("optimal_value") is not None:
                st.info(f"Giá trị tối ưu (z) = {fmt(result['optimal_value'])}")
            
            if result.get("solution") is not None:
                sol_str = ", ".join([f"{k} = {fmt(v)}" for k, v in result["solution"].items()])
                st.success(f"Nghiệm: {sol_str}")
        
        with st.expander("Xem chi tiết các bước giải (Nhấp để mở)", expanded=True):
            st.code(output, language="text")

with tab2:
    st.header("Phương pháp Hình học (Chỉ dùng cho 2 biến)")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Tạo ngẫu nhiên (Hình học)"):
            st.session_state["g_obj"] = random.choice(["max", "min"])
            st.session_state["g_bx1"] = random.choice([">=0", "<=0", "Tự do"])
            st.session_state["g_bx2"] = random.choice([">=0", "<=0", "Tự do"])
            st.session_state["g_c1"] = float(random.randint(-10, 10))
            st.session_state["g_c2"] = float(random.randint(-10, 10))
            n_cons_g = int(st.session_state.get("g_ncons", 2))
            for i in range(n_cons_g):
                st.session_state[f"ga1_{i}"] = float(random.randint(-10, 10))
                st.session_state[f"ga2_{i}"] = float(random.randint(-10, 10))
                st.session_state[f"grhs_{i}"] = float(random.randint(-20, 20))
                st.session_state[f"gsense_{i}"] = random.choice(["<=", ">=", "="])
                
    col1, col2 = st.columns(2)
    with col1:
        c1 = st.number_input("Hệ số mục tiêu x1", value=1.0, key="g_c1")
    with col2:
        c2 = st.number_input("Hệ số mục tiêu x2", value=1.0, key="g_c2")
        
    obj_sense_g = st.selectbox("Mục tiêu hình học", ["max", "min"], key="g_obj")
    
    st.subheader("Ràng buộc dấu (x1, x2)")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        bx1_display = st.selectbox("Dấu x1", [">=0", "<=0", "Tự do"], key="g_bx1")
        bx1 = BOUNDS_MAP[bx1_display]
    with bcol2:
        bx2_display = st.selectbox("Dấu x2", [">=0", "<=0", "Tự do"], key="g_bx2")
        bx2 = BOUNDS_MAP[bx2_display]
        
    num_cons_g = st.number_input("Số lượng ràng buộc hệ thống", min_value=1, value=2, step=1, key="g_ncons")
    constraints_g = []
    
    for i in range(int(num_cons_g)):
        st.markdown(f"**Ràng buộc {i+1}**")
        cols = st.columns(4)
        with cols[0]:
            a1 = st.number_input(f"Hệ số x1", value=1.0, key=f"ga1_{i}")
        with cols[1]:
            a2 = st.number_input(f"Hệ số x2", value=1.0, key=f"ga2_{i}")
        with cols[2]:
            sense = st.selectbox("Dấu", ["<=", ">=", "="], key=f"gsense_{i}")
        with cols[3]:
            rhs = st.number_input("RHS", value=10.0, key=f"grhs_{i}")
        constraints_g.append((a1, a2, sense, rhs))
        
    if st.button("Giải bằng Hình học", type="primary", key="g_solve"):
        cons_list = [Constraint2D(c[0], c[1], c[2], c[3], "") for c in constraints_g]
        temp_img_path = os.path.join(current_dir, "temp_graph.png")
        
        with st.spinner("Đang vẽ đồ thị..."):
            try:
                result = plot_lp_2d(
                    c=(c1, c2),
                    objective=obj_sense_g,
                    constraints=cons_list,
                    bounds=(bx1, bx2),
                    xlim=None,
                    ylim=None,
                    title="Đồ thị Hình học",
                    save_path=temp_img_path,
                    show=False
                )
                
                optimal_type = result.get("optimal_type", "")
                status_msg = result.get("status", "")
                if optimal_type == "single" or optimal_type == "multiple":
                    st.success(f"Trạng thái: {status_msg}")
                elif optimal_type == "none":
                    st.error(f"Trạng thái: {status_msg}")
                else:
                    st.info(f"Trạng thái: {status_msg}")

                if result.get("optimal_value") is not None:
                    st.info(f"Giá trị tối ưu (z) = {fmt_float(result['optimal_value'])}")
                    
                if os.path.exists(temp_img_path):
                    st.image(temp_img_path, use_column_width=True)
                    
            except Exception as e:
                st.error(f"Lỗi trong quá trình giải: {str(e)}")
