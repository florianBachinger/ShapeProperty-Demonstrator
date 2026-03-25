"""
Shape-Extractor Application
============================
Three-tab web application:
  Tab 1: Draw a curve -> symbolic regression (PyOperon) -> extract shape properties
  Tab 2: Review / edit / add shape constraints (LaTeX rendered)
  Tab 3: Draw a curve -> constrained P-spline fit (CPsplines) using constraints

Run:  python app.py
Open: http://localhost:8765/
"""

import asyncio
import json
import multiprocessing as mp
import signal
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bin_points(pts):
    """Bin raw drawn points into (X, y) arrays suitable for regression."""
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)

    n_bins = min(200, len(xs))
    x_min, x_max = xs.min(), xs.max()
    if x_max - x_min < 1e-9:
        return None, None, x_min, x_max
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(xs, bin_edges) - 1, 0, n_bins - 1)

    x_binned, y_binned = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.any():
            x_binned.append(xs[mask].mean())
            y_binned.append(ys[mask].mean())

    X = np.array(x_binned, dtype=np.float64)
    y = np.array(y_binned, dtype=np.float64)
    return X, y, x_min, x_max


# ---------------------------------------------------------------------------
# 1. Symbolic regression subprocess  (Tab 1)
# ---------------------------------------------------------------------------

def _run_regression(points_json: str, result_queue: mp.Queue):
    """Run symbolic regression in a child process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from pyoperon.sklearn import SymbolicRegressor
        import sympy as sy

        pts = json.loads(points_json)
        X, y, x_min, x_max = _bin_points(pts)
        if X is None:
            result_queue.put({"type": "error", "message": "Draw a wider curve"})
            return
        if len(X) < 3:
            result_queue.put({"type": "error", "message": "Need more points"})
            return

        Xr = X.reshape(-1, 1)

        params = {
            "allowed_symbols": "add,sub,mul,div,constant,variable",
            "population_size": 500,
            "pool_size": 500,
            "generations": 100,
            "female_selector": "tournament",
            "male_selector": "tournament",
            "tournament_size": 3,
            "optimizer_iterations": 10,
            "optimizer": "lm",
            "epsilon": 1e-05,
            "max_evaluations": 500000,
            "max_length": 50,
            "model_selection_criterion": "minimum_description_length",
            "mutation_probability": 0.15,
            "objectives": ["r2", "length"],
            "random_state": None,
            "uncertainty": [0.05],
            "n_threads": 0,
        }

        reg = SymbolicRegressor(**params)
        reg.fit(Xr, y)

        model = reg.model_
        expr_str = reg.get_model_string(model, 3)
        x_dense = np.linspace(x_min, x_max, 300)

        # --- helpers: scientific formatting --------------------------------
        def _sci_fmt(val, decimals=3):
            if val == 0:
                return "0"
            abs_val = abs(val)
            exp = int(np.floor(np.log10(abs_val)))
            if -2 <= exp <= 2:
                return f"{val:.{decimals}f}"
            mantissa = val / 10 ** exp
            return f"{mantissa:.{decimals}f}e{exp}"

        def _sci_latex_fmt(val, decimals=3):
            if val == 0:
                return "0"
            abs_val = abs(val)
            sign_str = "-" if val < 0 else ""
            exp = int(np.floor(np.log10(abs_val)))
            if -2 <= exp <= 2:
                return f"{val:.{decimals}f}"
            mantissa = abs_val / 10 ** exp
            return sign_str + f"{mantissa:.{decimals}f}" + r" \times 10^{" + str(exp) + "}"

        def _round_and_sci(expr):
            for atom in list(expr.atoms(sy.Number)):
                if isinstance(atom, sy.Float):
                    expr = expr.subs(atom, sy.Float(float(atom), 4))
            d = str(expr)
            for atom in sorted(expr.atoms(sy.Number), key=lambda a: -len(str(a))):
                if isinstance(atom, sy.Float):
                    d = d.replace(str(atom), _sci_fmt(float(atom)))
            l = sy.latex(expr)
            for atom in sorted(expr.atoms(sy.Number), key=lambda a: -len(sy.latex(a))):
                if isinstance(atom, sy.Float):
                    l = l.replace(sy.latex(atom), _sci_latex_fmt(float(atom)))
            return d, l

        def _process_tree(tree_str):
            try:
                parsed = sy.parse_expr(tree_str.lower())
                simplified = sy.simplify(parsed)
                d, l = _round_and_sci(simplified)
                x_sym = sy.Symbol('x0') if 'x0' in tree_str.lower() else (
                    sy.Symbol('x_0') if 'x_0' in tree_str.lower() else
                    list(simplified.free_symbols)[0] if simplified.free_symbols else None
                )
                if x_sym is not None:
                    fn = sy.lambdify(x_sym, simplified, modules='numpy')
                    y_vals = fn(x_dense)
                    y_arr = np.atleast_1d(np.asarray(y_vals, dtype=np.float64))
                    if y_arr.shape == ():
                        y_arr = np.full_like(x_dense, float(y_arr))
                    if y_arr.shape[0] == 1 and x_dense.shape[0] > 1:
                        y_arr = np.full_like(x_dense, y_arr[0])
                    crv = list(zip(x_dense.tolist(), y_arr.tolist()))
                else:
                    c = float(simplified)
                    crv = list(zip(x_dense.tolist(), np.full_like(x_dense, c).tolist()))
                return d, l, crv
            except Exception:
                return tree_str, tree_str, []

        display_str, latex_str, best_curve = _process_tree(expr_str)

        try:
            y_pred = reg.predict(x_dense.reshape(-1, 1))
            best_curve = list(zip(x_dense.tolist(), y_pred.tolist()))
        except Exception:
            pass

        # --- pareto front ---------------------------------------------------
        pareto = []
        best_idx = 0
        for i, s in enumerate(reg.pareto_front_):
            obj = s["objective_values"]
            tree = s["tree"]
            mdl = s["minimum_description_length"]
            ts = reg.get_model_string(tree, 3)
            p_display, p_latex, p_curve = _process_tree(ts)
            if p_display == display_str:
                best_idx = i
            pareto.append({
                "r2": (-1) * round(float(obj[0]), 6),
                "length": int(obj[1]),
                "mdl": round(float(mdl), 4),
                "expr": p_display,
                "latex": p_latex,
                "curve": p_curve,
            })

        result_queue.put({
            "type": "result",
            "expression": display_str,
            "latex": latex_str,
            "curve": best_curve,
            "pareto": pareto,
            "bestIdx": best_idx,
        })

    except Exception as e:
        result_queue.put({"type": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# 2. Shape extraction subprocess  (Tab 1 -> Tab 2)
# ---------------------------------------------------------------------------

def _extract_constraints(expr_str: str, x_min: float, x_max: float,
                         result_queue: mp.Queue):
    """
    Analyse a symbolic expression and extract shape constraints.

    For the function value, 1st derivative, and 2nd derivative:
      - Evaluate on a dense grid
      - If all signs agree -> global constraint
      - If mixed -> fit a shallow DecisionTree to find sub-domain splits
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import sympy as sy
        from sklearn.tree import DecisionTreeClassifier

        parsed = sy.parse_expr(expr_str.lower())
        simplified = sy.simplify(parsed)

        x_sym = sy.Symbol('x0') if 'x0' in expr_str.lower() else (
            sy.Symbol('x_0') if 'x_0' in expr_str.lower() else
            list(simplified.free_symbols)[0] if simplified.free_symbols else None
        )

        if x_sym is None:
            result_queue.put({"type": "constraints", "constraints": []})
            return

        f = simplified
        f_prime = sy.diff(f, x_sym)
        f_double_prime = sy.diff(f_prime, x_sym)

        n_pts = 10000
        x_vals = np.linspace(x_min, x_max, n_pts)
        X_df = x_vals.reshape(-1, 1)

        derivatives = [
            (f, 0, "f(x)"),
            (f_prime, 1, "f'(x)"),
            (f_double_prime, 2, "f''(x)"),
        ]

        constraints = []
        constraint_id = 0

        for expr, order, label in derivatives:
            try:
                fn = sy.lambdify(x_sym, expr, modules='numpy')
                vals = np.atleast_1d(np.asarray(fn(x_vals), dtype=np.float64))
                if vals.shape[0] != n_pts:
                    vals = np.full(n_pts, float(vals[0]))
            except Exception:
                continue

            vals = np.where(np.isfinite(vals), vals, 0.0)
            signs = np.sign(vals)

            nonzero_mask = signs != 0
            signs_nz = signs[nonzero_mask]
            x_nz = X_df[nonzero_mask]

            if len(signs_nz) == 0:
                continue

            unique_signs = np.unique(signs_nz)

            if len(unique_signs) == 1:
                sign_str = "+" if unique_signs[0] == 1 else "-"
                c = _make_constraint(
                    constraint_id, order, sign_str,
                    round(float(x_min), 4), round(float(x_max), 4)
                )
                constraints.append(c)
                constraint_id += 1
            else:
                clf = DecisionTreeClassifier(max_depth=3)
                clf.fit(x_nz, signs_nz)
                score = clf.score(x_nz, signs_nz)

                if score >= 0.995:
                    bounds = {0: (x_min, x_max)}
                    sub_constraints = _tree_to_constraints_1d(clf, 0, bounds)
                    for (rng, sign_str) in sub_constraints:
                        lo = round(float(rng[0][0]), 4)
                        hi = round(float(rng[0][1]), 4)
                        c = _make_constraint(constraint_id, order, sign_str, lo, hi)
                        constraints.append(c)
                        constraint_id += 1

        result_queue.put({"type": "constraints", "constraints": constraints})

    except Exception as e:
        result_queue.put({"type": "error", "message": f"Extraction failed: {e}"})


def _make_constraint(cid, order, sign, x_low, x_high):
    """Build a single constraint dict with LaTeX representation."""
    if order == 0:
        if sign == "+":
            desc = "non-negative"
            math_rhs = r"f(x) \geq 0"
        else:
            desc = "non-positive"
            math_rhs = r"f(x) \leq 0"
    elif order == 1:
        if sign == "+":
            desc = "monotonic non-decreasing"
            math_rhs = r"f'(x) \geq 0"
        else:
            desc = "monotonic non-increasing"
            math_rhs = r"f'(x) \leq 0"
    elif order == 2:
        if sign == "+":
            desc = "convex"
            math_rhs = r"f''(x) \geq 0"
        else:
            desc = "concave"
            math_rhs = r"f''(x) \leq 0"
    else:
        desc = f"order-{order} {'positive' if sign == '+' else 'negative'}"
        math_rhs = f"f^{{({order})}}(x) {'\\geq' if sign == '+' else '\\leq'} 0"

    latex = f"{x_low} \\leq x \\leq {x_high} \\implies {math_rhs}"

    return {
        "id": cid,
        "order": order,
        "sign": sign,
        "xLow": x_low,
        "xHigh": x_high,
        "description": desc,
        "latex": latex,
        "enabled": True,
    }


def _tree_to_constraints_1d(clf, node, bounds):
    """Recursively extract (bounds, sign) pairs from a 1D decision tree."""
    tree = clf.tree_
    if tree.children_left[node] == -1:
        class_counts = tree.value[node][0]
        if class_counts[0] >= class_counts[1]:
            return [(bounds, "-")]
        else:
            return [(bounds, "+")]

    feature = tree.feature[node]
    threshold = tree.threshold[node]

    left_bounds = dict(bounds)
    right_bounds = dict(bounds)
    lo, hi = bounds[feature]
    left_bounds[feature] = (lo, threshold)
    right_bounds[feature] = (threshold, hi)

    left = _tree_to_constraints_1d(clf, tree.children_left[node], left_bounds)
    right = _tree_to_constraints_1d(clf, tree.children_right[node], right_bounds)
    return left + right


# ---------------------------------------------------------------------------
# 3. CPsplines fitting subprocess  (Tab 3)
# ---------------------------------------------------------------------------

def _run_cpsplines(points_json: str, constraints_json: str,
                   result_queue: mp.Queue):
    """
    Fit the drawn curve with CPsplines, enforcing global shape constraints.

    Only constraints spanning the full domain are applied as a single fit.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import pandas as pd
        from cpsplines.fittings.fit_cpsplines import CPsplines

        pts = json.loads(points_json)
        constraints = json.loads(constraints_json)
        X, y, x_min, x_max = _bin_points(pts)
        if X is None:
            result_queue.put({"type": "error", "message": "Draw a wider curve"})
            return
        if len(X) < 3:
            result_queue.put({"type": "error", "message": "Need more points"})
            return

        enabled = [c for c in constraints if c.get("enabled", True)]

        data = pd.DataFrame({"x": X, "y": y})
        # Predict strictly within the fitted data range to stay inside
        # the B-spline basis definition domain.
        x_lo, x_hi = float(X.min()), float(X.max())
        x_dense = np.linspace(-10, 10, 300)

        # Build shape constraints dict from enabled (global) constraints
        shape_dict = {}
        constraint_descs = []
        for c in enabled:
            order = c["order"]
            sign_key = c["sign"]
            if order not in shape_dict:
                shape_dict[order] = {}
            shape_dict[order][sign_key] = 0
            constraint_descs.append(c["description"])

        shape_arg = {"x": shape_dict} if shape_dict else None

        # Fallback chain: constrained → unconstrained
        for attempt_shape, label in [
            (shape_arg, constraint_descs),
            (None, ["(unconstrained fallback)"]),
        ]:
            try:
                m = CPsplines(
                    deg=(3,), ord_d=(2,), k=(30,),
                    x_range={"x": (-10, 10)},
                    sp_args={"options": {"ftol": 1e-12}},
                    shape_constraints=attempt_shape,
                )
                m.fit(data=data, y_col="y")
                y_pred = m.predict(pd.DataFrame({"x": x_dense}))
                curve = list(zip(x_dense.tolist(), y_pred.tolist()))
                result_queue.put({
                    "type": "cpsplines_result",
                    "curve": curve,
                    "constraints_applied": label,
                })
                return
            except Exception:
                continue

        result_queue.put({"type": "error", "message": "CPsplines fit failed"})

    except Exception as e:
        result_queue.put({"type": "error", "message": f"CPsplines failed: {e}"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "templates" / "index.html").read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    process: mp.Process | None = None
    result_queue: mp.Queue | None = None

    def _kill_process():
        nonlocal process, result_queue
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        process = None
        result_queue = None

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.1
                )
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                msg = None
            except WebSocketDisconnect:
                break

            if msg is not None:
                if msg["type"] == "cancel":
                    _kill_process()

                elif msg["type"] == "fit":
                    _kill_process()
                    points = msg["points"]
                    result_queue = mp.Queue()
                    process = mp.Process(
                        target=_run_regression,
                        args=(json.dumps(points), result_queue),
                        daemon=True,
                    )
                    process.start()

                elif msg["type"] == "extract":
                    _kill_process()
                    expr_str = msg["expression"]
                    x_min = msg.get("xMin", -10)
                    x_max = msg.get("xMax", 10)
                    result_queue = mp.Queue()
                    process = mp.Process(
                        target=_extract_constraints,
                        args=(expr_str, x_min, x_max, result_queue),
                        daemon=True,
                    )
                    process.start()

                elif msg["type"] == "fit_cpsplines":
                    _kill_process()
                    points = msg["points"]
                    constraints_data = msg["constraints"]
                    result_queue = mp.Queue()
                    process = mp.Process(
                        target=_run_cpsplines,
                        args=(
                            json.dumps(points),
                            json.dumps(constraints_data),
                            result_queue,
                        ),
                        daemon=True,
                    )
                    process.start()

            # Poll result queue
            if result_queue is not None:
                try:
                    result = result_queue.get_nowait()
                    await websocket.send_text(json.dumps(result))
                    if process is not None:
                        process.join(timeout=1)
                    process = None
                    result_queue = None
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        _kill_process()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    uvicorn.run(app, host="0.0.0.0", port=8765)
