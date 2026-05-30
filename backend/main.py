from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
import io
import contextlib
import os
import sys

# Thêm đường dẫn project vào sys.path để import được Simplex
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from Simplex.simplex_dictionary_solver import SimplexDictionarySolver, Constraint

app = FastAPI(title="Quy Hoạch Tuyến Tính API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConstraintModel(BaseModel):
    coeffs: List[float]
    sense: Literal["<=", ">=", "="]
    rhs: float

class SimplexRequest(BaseModel):
    objective_coeffs: List[float]
    objective_sense: Literal["max", "min"]
    constraints: List[ConstraintModel]
    pivot_rule: Literal["dantzig", "bland"] = "bland"
    
@app.post("/api/solve/simplex")
def solve_simplex(req: SimplexRequest):
    num_vars = len(req.objective_coeffs)
    bounds = [">=0"] * num_vars
    var_names = [f"x{i+1}" for i in range(num_vars)]
    
    cons_list = []
    for c in req.constraints:
        cons_list.append(Constraint(c.coeffs, c.sense, c.rhs))
        
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        solver = SimplexDictionarySolver(
            c=req.objective_coeffs,
            constraints=cons_list,
            objective=req.objective_sense,
            bounds=bounds,
            var_names=var_names,
            pivot_rule=req.pivot_rule,
            verbose=True
        )
        result = solver.solve()
        
    steps_text = f.getvalue()
    
    solution_dict = {}
    if result["solution"] is not None:
        solution_dict = {k: float(v) for k, v in result["solution"].items()}
        
    opt_val = float(result["optimal_value"]) if result["optimal_value"] is not None else None
    
    return {
        "status": result["status"],
        "optimal_value": opt_val,
        "solution": solution_dict,
        "steps": steps_text
    }

from Graphical.graphical_lp_2d import plot_lp_2d, Constraint2D
import base64

class GraphicalConstraintModel(BaseModel):
    a1: float
    a2: float
    sense: Literal["<=", ">=", "="]
    rhs: float

class GraphicalRequest(BaseModel):
    c1: float
    c2: float
    objective_sense: Literal["max", "min"]
    constraints: List[GraphicalConstraintModel]

@app.post("/api/solve/graphical")
def solve_graphical(req: GraphicalRequest):
    cons_list = []
    for c in req.constraints:
        cons_list.append(Constraint2D(c.a1, c.a2, c.sense, c.rhs, ""))
        
    try:
        temp_img_path = os.path.join(project_dir, "temp_graph.png")
        result = plot_lp_2d(
            c=(req.c1, req.c2),
            objective=req.objective_sense,
            constraints=cons_list,
            bounds=(">=0", ">=0"),
            xlim=(-1, 15),
            ylim=(-1, 15),
            title="Đồ thị Hình học",
            save_path=temp_img_path,
            show=False
        )
        
        # Đọc ảnh và chuyển sang base64
        image_b64 = ""
        if os.path.exists(temp_img_path):
            with open(temp_img_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        opt_val = float(result["optimal_value"]) if result.get("optimal_value") is not None else None
        
        return {
            "status": result["status"],
            "optimal_value": opt_val,
            "image_base64": image_b64
        }
    except Exception as e:
        return {"error": str(e)}

# Phục vụ file tĩnh tĩnh Frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_path = os.path.join(project_dir, "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{filename}")
    def serve_frontend_files(filename: str):
        file_path = os.path.join(frontend_path, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        return {"error": "File not found"}
