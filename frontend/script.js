document.addEventListener('DOMContentLoaded', () => {
    renderSimplexInputs();
    renderGraphicalInputs();
});

// ================= SIMPLEX LOGIC =================
function renderSimplexInputs() {
    const numVars = parseInt(document.getElementById('numVarsS').value);
    const numCons = parseInt(document.getElementById('numConsS').value);
    
    // Hàm mục tiêu
    const objContainer = document.getElementById('objInputsS');
    objContainer.innerHTML = '<span class="fw-bold fs-5">Z = </span>';
    for (let i = 1; i <= numVars; i++) {
        objContainer.innerHTML += `<input type="number" class="form-control input-number" id="s_c_${i}" value="0"> <span class="math-var">x<sub>${i}</sub></span>`;
        if (i < numVars) objContainer.innerHTML += '<span class="fw-bold">+</span>';
    }
    
    // Ràng buộc
    const conContainer = document.getElementById('conInputsS');
    conContainer.innerHTML = '';
    for (let i = 1; i <= numCons; i++) {
        let rowHtml = `<div class="d-flex align-items-center gap-2 flex-wrap">`;
        for (let j = 1; j <= numVars; j++) {
            rowHtml += `<input type="number" class="form-control input-number" id="s_con_${i}_x_${j}" value="0"> <span class="math-var">x<sub>${j}</sub></span>`;
            if (j < numVars) rowHtml += '<span class="fw-bold">+</span>';
        }
        rowHtml += `
            <select class="form-select w-auto fw-bold" id="s_con_${i}_sense">
                <option value="<=">&le;</option>
                <option value=">=">&ge;</option>
                <option value="=">=</option>
            </select>
            <input type="number" class="form-control input-number" id="s_con_${i}_rhs" value="0">
        </div>`;
        conContainer.innerHTML += rowHtml;
    }
}

function randomizeSimplex() {
    const numVars = parseInt(document.getElementById('numVarsS').value);
    const numCons = parseInt(document.getElementById('numConsS').value);
    
    for (let i = 1; i <= numVars; i++) {
        document.getElementById(`s_c_${i}`).value = Math.floor(Math.random() * 21) - 10;
    }
    for (let i = 1; i <= numCons; i++) {
        for (let j = 1; j <= numVars; j++) {
            document.getElementById(`s_con_${i}_x_${j}`).value = Math.floor(Math.random() * 21) - 10;
        }
        document.getElementById(`s_con_${i}_rhs`).value = Math.floor(Math.random() * 50) + 1;
        document.getElementById(`s_con_${i}_sense`).value = '<=';
    }
}

async function solveSimplex() {
    const btn = document.getElementById('btnSolveSimplex');
    btn.innerHTML = '⏳ Đang giải...';
    btn.disabled = true;
    
    try {
        const numVars = parseInt(document.getElementById('numVarsS').value);
        const numCons = parseInt(document.getElementById('numConsS').value);
        const c = [];
        for (let i = 1; i <= numVars; i++) c.push(parseFloat(document.getElementById(`s_c_${i}`).value));
        
        const constraints = [];
        for (let i = 1; i <= numCons; i++) {
            const row = [];
            for (let j = 1; j <= numVars; j++) row.push(parseFloat(document.getElementById(`s_con_${i}_x_${j}`).value));
            constraints.push({
                coeffs: row,
                sense: document.getElementById(`s_con_${i}_sense`).value,
                rhs: parseFloat(document.getElementById(`s_con_${i}_rhs`).value)
            });
        }
        
        const payload = {
            objective_coeffs: c,
            objective_sense: document.getElementById('objSenseS').value,
            constraints: constraints,
            pivot_rule: document.getElementById('pivotRuleS').value
        };
        
        const response = await fetch('/api/solve/simplex', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        document.getElementById('resultSectionS').style.display = 'block';
        const { displayStatus, displayOptVal } = formatResultStatus(data.status, document.getElementById('objSenseS').value, data.optimal_value);
        
        document.getElementById('statusTextS').innerText = displayStatus;
        document.getElementById('optValueTextS').innerText = displayOptVal;
        
        // Parse the steps
        const stepsContainer = document.getElementById('stepsContainerS');
        stepsContainer.innerHTML = '';
        const chunks = parseSteps(data.steps);
        for (let chunk of chunks) {
            const linesArr = chunk.trim().split('\n');
            const firstLine = linesArr[0];
            const rest = linesArr.slice(1).join('\n');
            stepsContainer.innerHTML += `
                <div class="step-box">
                    <div class="step-title">${firstLine}</div>
                    <div class="step-content">${rest}</div>
                </div>`;
        }
        document.getElementById('resultSectionS').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        alert("Lỗi: " + e.message);
    } finally {
        btn.innerHTML = '🚀 Giải Simplex';
        btn.disabled = false;
    }
}

function parseSteps(stepsText) {
    const lines = stepsText.split('\n');
    let chunks = [];
    let currentChunk = [];
    for (let line of lines) {
        if (line.startsWith('DẠNG CHUẨN') || line.startsWith('Từ vựng') || line.startsWith('[Pha ') || line.startsWith('Loại x0') || line.startsWith('KẾT LUẬN')) {
            if (currentChunk.length > 0) chunks.push(currentChunk.join('\n'));
            currentChunk = [];
        }
        if (line.trim() !== '' || currentChunk.length > 0) currentChunk.push(line);
    }
    if (currentChunk.length > 0) chunks.push(currentChunk.join('\n'));
    return chunks;
}


// ================= GRAPHICAL LOGIC =================
function renderGraphicalInputs() {
    const numCons = parseInt(document.getElementById('numConsG').value);
    
    const conContainer = document.getElementById('conInputsG');
    conContainer.innerHTML = '';
    for (let i = 1; i <= numCons; i++) {
        conContainer.innerHTML += `
        <div class="d-flex align-items-center gap-2 flex-wrap">
            <input type="number" class="form-control input-number" id="g_con_${i}_x1" value="0"> <span class="math-var">x<sub>1</sub></span>
            <span class="fw-bold">+</span>
            <input type="number" class="form-control input-number" id="g_con_${i}_x2" value="0"> <span class="math-var">x<sub>2</sub></span>
            <select class="form-select w-auto fw-bold" id="g_con_${i}_sense">
                <option value="<=">&le;</option>
                <option value=">=">&ge;</option>
                <option value="=">=</option>
            </select>
            <input type="number" class="form-control input-number" id="g_con_${i}_rhs" value="0">
        </div>`;
    }
}

function randomizeGraphical() {
    const numCons = parseInt(document.getElementById('numConsG').value);
    document.getElementById('g_c1').value = Math.floor(Math.random() * 21) - 10;
    document.getElementById('g_c2').value = Math.floor(Math.random() * 21) - 10;
    
    for (let i = 1; i <= numCons; i++) {
        document.getElementById(`g_con_${i}_x1`).value = Math.floor(Math.random() * 21) - 10;
        document.getElementById(`g_con_${i}_x2`).value = Math.floor(Math.random() * 21) - 10;
        document.getElementById(`g_con_${i}_rhs`).value = Math.floor(Math.random() * 30) + 1;
        document.getElementById(`g_con_${i}_sense`).value = '<=';
    }
}

async function solveGraphical() {
    const btn = document.getElementById('btnSolveGraphical');
    btn.innerHTML = '⏳ Đang vẽ...';
    btn.disabled = true;
    
    try {
        const numCons = parseInt(document.getElementById('numConsG').value);
        const c1 = parseFloat(document.getElementById('g_c1').value);
        const c2 = parseFloat(document.getElementById('g_c2').value);
        
        const constraints = [];
        for (let i = 1; i <= numCons; i++) {
            constraints.push({
                a1: parseFloat(document.getElementById(`g_con_${i}_x1`).value),
                a2: parseFloat(document.getElementById(`g_con_${i}_x2`).value),
                sense: document.getElementById(`g_con_${i}_sense`).value,
                rhs: parseFloat(document.getElementById(`g_con_${i}_rhs`).value)
            });
        }
        
        const payload = {
            c1: c1,
            c2: c2,
            objective_sense: document.getElementById('objSenseG').value,
            constraints: constraints
        };
        
        const response = await fetch('/api/solve/graphical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        if (data.error) {
            alert("Lỗi từ server: " + data.error);
            return;
        }
        
        document.getElementById('resultSectionG').style.display = 'block';
        const { displayStatus, displayOptVal } = formatResultStatus(data.status, document.getElementById('objSenseG').value, data.optimal_value);
        
        document.getElementById('statusTextG').innerText = displayStatus;
        document.getElementById('optValueTextG').innerText = displayOptVal;
        
        // Show image
        const img = document.getElementById('graphImage');
        img.src = 'data:image/png;base64,' + data.image_base64;
        img.style.display = 'block';
        
        document.getElementById('resultSectionG').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        alert("Lỗi: " + e.message);
    } finally {
        btn.innerHTML = '📈 Vẽ Đồ Thị';
        btn.disabled = false;
    }
}

function formatResultStatus(statusStr, objSense, optValue) {
    let displayStatus = "";
    let displayOptVal = "";
    
    // Normalize status string
    const s = statusStr.toLowerCase();
    
    if (s.includes("optimal") || s.includes("có nghiệm") || s.includes("vô số nghiệm") || s === "tối ưu") {
        displayStatus = "Tối ưu";
        displayOptVal = 'Giá trị tối ưu (Z) = ' + (optValue !== null ? optValue : '?');
    } else if (s.includes("unbounded") || s.includes("không giới nội")) {
        displayStatus = "Không giới nội";
        displayOptVal = (objSense === "min") ? "min z = -∞" : "max z = +∞";
    } else if (s.includes("infeasible") || s.includes("vô nghiệm") || s.includes("không tìm được đỉnh khả thi")) {
        displayStatus = "Vô nghiệm";
        displayOptVal = (objSense === "min") ? "min z = +∞" : "max z = -∞";
    } else {
        displayStatus = statusStr;
        displayOptVal = (optValue !== null) ? 'Giá trị tối ưu (Z) = ' + optValue : '';
    }
    
    return { displayStatus, displayOptVal };
}
