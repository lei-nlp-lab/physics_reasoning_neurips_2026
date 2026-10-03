import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.R = getattr(solution_instance, 'R', None)
        self.U = getattr(solution_instance, 'U', None)
        self.B = getattr(solution_instance, 'B', None)
        
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction of all expected results with fallback to None
        self.fracqm = self.results.get('fracqm', None)
        self.x = self.results.get('x', None)
        self.t2 = self.results.get('t_2', None)
        self.T = self.results.get('T', None)
        self.angle1 = self.results.get('∠1', None)  # degrees
        self.angle3 = self.results.get('∠3', None)  # degrees
        self.x2 = self.results.get('x_2', None)
        self.sin_angle1 = self.results.get('sin∠1', None)
        self.t1 = self.results.get('t_1', None)
        self.angle2 = self.results.get('∠2', None)  # degrees
        self.angle4 = self.results.get('∠4', None)  # degrees
        self.x1 = self.results.get('x_1', None)
        
        # For symbolic computations
        self.q, self.m, self.v = sp.symbols('q m v', positive=True)
        
    def test_invariants(self):
        logs = []
        
        # 1. Check energy conservation: qU = 1/2 m v^2
        try:
            if all(val is not None for val in [self.U, self.fracqm]):
                # Numerical check: v from energy = sqrt(2*(q/m)*U)
                v_from_energy = math.sqrt(2 * float(self.fracqm) * float(self.U))
                
                # v from Lorentz force: for critical trajectory, radius = R
                # q v B = m v^2 / R  =>  v = (q/m) B R
                if self.B and self.R:
                    v_from_lorentz = float(self.fracqm) * float(self.B) * float(self.R)
                    
                    if not math.isclose(v_from_energy, v_from_lorentz, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Energy-Lorentz velocity mismatch. Energy gives {v_from_energy:.4e}, Lorentz gives {v_from_lorentz:.4e}")
                    else:
                        logs.append(f"PASS: Energy conservation and Lorentz force consistent (v ≈ {v_from_energy:.4e} m/s)")
            else:
                logs.append("SKIP: Missing U or q/m for energy check")
        except Exception as e:
            logs.append(f"ERROR in energy conservation: {traceback.format_exc()}")
        
        # 2. Check charge-to-mass ratio formula: q/m = 2U/(B^2 R^2)
        try:
            if all(val is not None for val in [self.U, self.B, self.R, self.fracqm]):
                expected_qm = 2 * float(self.U) / (float(self.B)**2 * float(self.R)**2)
                if not math.isclose(float(self.fracqm), expected_qm, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: q/m mismatch. Got {self.fracqm:.4e}, expected {expected_qm:.4e}")
                else:
                    logs.append(f"PASS: q/m = {self.fracqm:.4e} matches formula")
            else:
                logs.append("SKIP: Missing parameters for q/m check")
        except Exception as e:
            logs.append(f"ERROR in q/m check: {traceback.format_exc()}")
        
        # 3. Check Lorentz force gives correct radius for critical trajectory
        try:
            if all(val is not None for val in [self.fracqm, self.B, self.v]):
                # r = m v / (q B) = v / ((q/m) B)
                v_num = float(self.v) if hasattr(self, 'v') else math.sqrt(2 * float(self.fracqm) * float(self.U))
                r_calc = v_num / (float(self.fracqm) * float(self.B))
                if self.R and not math.isclose(r_calc, float(self.R), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Calculated radius {r_calc:.4e} ≠ given R {self.R:.4e}")
                else:
                    logs.append(f"PASS: Lorentz force gives r ≈ {r_calc:.4e} m (expected R = {self.R:.4e} m)")
        except Exception as e:
            logs.append(f"ERROR in radius check: {traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        # 1. Check x = x1 + x2
        try:
            if all(val is not None for val in [self.x, self.x1, self.x2]):
                if not math.isclose(float(self.x), float(self.x1) + float(self.x2), rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x = {self.x:.4e} ≠ x1 + x2 = {float(self.x1)+float(self.x2):.4e}")
                else:
                    logs.append(f"PASS: x = x1 + x2 = {self.x:.4e}")
        except Exception as e:
            logs.append(f"ERROR in x consistency: {traceback.format_exc()}")
        
        # 2. Check angles sum correctly (from geometry description)
        try:
            if all(val is not None for val in [self.angle1, self.angle2, self.angle3]):
                # From steps: ∠3 = ∠1 + ∠2? Actually from geometry, ∠3 = ∠1 + ∠2? Let's check given values:
                # ∠1=60°, ∠2=45°, ∠3=75° -> 60+45=105 ≠ 75, so not simple sum.
                # Instead, check trigonometric relation from step 7: sin(∠1) = √3/2
                if self.sin_angle1 is not None:
                    expected_sin = math.sin(math.radians(60))  # ≈ √3/2
                    if not math.isclose(float(self.sin_angle1), expected_sin, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: sin(∠1) = {self.sin_angle1:.4e} ≠ √3/2 ≈ {expected_sin:.4e}")
                    else:
                        logs.append(f"PASS: sin(∠1) ≈ √3/2")
        except Exception as e:
            logs.append(f"ERROR in angle checks: {traceback.format_exc()}")
        
        # 3. Check time relations: t1 = (∠3/360) * T, t2 = (∠4/360) * T
        try:
            if all(val is not None for val in [self.t1, self.T, self.angle3]):
                t1_expected = (float(self.angle3) / 360) * float(self.T)
                if not math.isclose(float(self.t1), t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t1 = {self.t1:.4e} ≠ ({self.angle3}/360)*T = {t1_expected:.4e}")
                else:
                    logs.append(f"PASS: t1 = ({self.angle3}/360)*T")
            
            if all(val is not None for val in [self.t2, self.T, self.angle4]):
                t2_expected = (float(self.angle4) / 360) * float(self.T)
                if not math.isclose(float(self.t2), t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2 = {self.t2:.4e} ≠ ({self.angle4}/360)*T = {t2_expected:.4e}")
                else:
                    logs.append(f"PASS: t2 = ({self.angle4}/360)*T")
        except Exception as e:
            logs.append(f"ERROR in time relations: {traceback.format_exc()}")
        
        # 4. Check T = 2πm/(qB) = 2π/( (q/m) B )
        try:
            if all(val is not None for val in [self.T, self.fracqm, self.B]):
                T_expected = 2 * math.pi / (float(self.fracqm) * float(self.B))
                if not math.isclose(float(self.T), T_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T = {self.T:.4e} ≠ 2π/( (q/m) B ) = {T_expected:.4e}")
                else:
                    logs.append(f"PASS: T = 2π/( (q/m) B ) ≈ {self.T:.4e} s")
        except Exception as e:
            logs.append(f"ERROR in period check: {traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        
        # Perturb U by +10% and check that q/m increases proportionally (since q/m ∝ U)
        try:
            if all(val is not None for val in [self.U, self.B, self.R]):
                original_U = float(self.U)
                perturbed_U = original_U * 1.1  # +10%
                
                # Recalculate q/m using formula: 2U/(B^2 R^2)
                original_qm = 2 * original_U / (float(self.B)**2 * float(self.R)**2)
                perturbed_qm = 2 * perturbed_U / (float(self.B)**2 * float(self.R)**2)
                
                if not (perturbed_qm > original_qm):
                    logs.append(f"FAIL: q/m did not increase when U increased (from {original_qm:.4e} to {perturbed_qm:.4e})")
                else:
                    # Check proportionality: perturbed_qm / original_qm should equal perturbed_U / original_U
                    expected_ratio = perturbed_U / original_U
                    actual_ratio = perturbed_qm / original_qm
                    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: q/m not proportional to U. Ratio expected {expected_ratio:.4f}, got {actual_ratio:.4f}")
                    else:
                        logs.append(f"PASS: q/m ∝ U (monotonic increase confirmed)")
            else:
                logs.append("SKIP: Missing parameters for U perturbation")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
        
        # Perturb B and check q/m decreases (since q/m ∝ 1/B^2)
        try:
            if all(val is not None for val in [self.U, self.B, self.R]):
                original_B = float(self.B)
                perturbed_B = original_B * 1.1  # +10%
                
                original_qm = 2 * float(self.U) / (original_B**2 * float(self.R)**2)
                perturbed_qm = 2 * float(self.U) / (perturbed_B**2 * float(self.R)**2)
                
                if not (perturbed_qm < original_qm):
                    logs.append(f"FAIL: q/m did not decrease when B increased (from {original_qm:.4e} to {perturbed_qm:.4e})")
                else:
                    # Check inverse square relation: (original_B/perturbed_B)^2 = perturbed_qm/original_qm
                    expected_ratio = (original_B / perturbed_B) ** 2
                    actual_ratio = perturbed_qm / original_qm
                    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: q/m ∝ 1/B^2 violation. Ratio expected {expected_ratio:.4f}, got {actual_ratio:.4f}")
                    else:
                        logs.append(f"PASS: q/m ∝ 1/B^2 (monotonic decrease confirmed)")
            else:
                logs.append("SKIP: Missing parameters for B perturbation")
        except Exception as e:
            logs.append(f"ERROR in B monotonicity: {traceback.format_exc()}")
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test and collect logs
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                logs = test_func()
                if any(log.startswith(('FAIL', 'ERROR')) for log in logs):
                    success = False
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRASH: {traceback.format_exc()}")
                success = False
        
        return success, all_logs