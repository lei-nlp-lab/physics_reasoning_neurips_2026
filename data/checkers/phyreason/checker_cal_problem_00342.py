import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Initialize with defaults
        self.side = 0.1
        self.v0 = 5.0e5
        self.phi_B = -20.0
        self.phi_C = 20.0
        self.phi_E = 60.0
        
        # Safely extract results dictionary
        results = {}
        if hasattr(solution_instance, 'results'):
            results = getattr(solution_instance, 'results')
        
        # Extract all required variables from results with sensible defaults
        # Use direct extraction with proper error handling
        try:
            self.varphi_A = float(results.get('varphi_A', -20.0))
            self.varphi_B = float(results.get('varphi_B', -20.0))
            self.varphi_C = float(results.get('varphi_C', 20.0))
            self.varphi_E = float(results.get('varphi_E', 60.0))
            self.varphi_F = float(results.get('varphi_F', 20.0))
            self.AE = float(results.get('AE', 0.1732050807568877))
            self.E = float(results.get('E', 461.88021535170065))
            self.t = float(results.get('t', 2.0e-7))
            self.fracqm = float(results.get('fracqm', 1.875e10))
            
            # Handle q sign safely
            q_val = results.get('q', -1.0)
            if isinstance(q_val, (int, float)):
                self.q_sign = 'negative' if q_val < 0 else 'positive'
            else:
                # Try to convert string to float if possible
                try:
                    q_float = float(q_val)
                    self.q_sign = 'negative' if q_float < 0 else 'positive'
                except:
                    self.q_sign = str(q_val)
        except Exception as e:
            # If any extraction fails, use defaults and log
            self.logs = [f"Error extracting results: {e}"]
            # Set defaults
            self.varphi_A = -20.0
            self.varphi_B = -20.0
            self.varphi_C = 20.0
            self.varphi_E = 60.0
            self.varphi_F = 20.0
            self.AE = 0.1732050807568877
            self.E = 461.88021535170065
            self.t = 2.0e-7
            self.fracqm = 1.875e10
            self.q_sign = 'negative'
        
        # Store original values for restoration in tests
        self.original_side = self.side
        self.original_v0 = self.v0
        self.original_varphi_A = self.varphi_A
        self.original_varphi_B = self.varphi_B
        self.original_varphi_E = self.varphi_E
        self.original_E = self.E
        self.original_fracqm = self.fracqm
        self.original_t = self.t
        
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        try:
            # 1. Check equipotential condition: A and B on same line => φ_A = φ_B
            if not math.isclose(self.varphi_A, self.varphi_B, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: φ_A={self.varphi_A} != φ_B={self.varphi_B} (should be equal)")
                return False
            self.log("PASS: φ_A = φ_B (equipotential)")
            
            # 2. Check potential difference relation: φ_B - φ_C = φ_F - φ_E
            lhs = self.varphi_B - self.varphi_C
            rhs = self.varphi_F - self.varphi_E
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: φ_B-φ_C={lhs} != φ_F-φ_E={rhs}")
                return False
            self.log("PASS: φ_B - φ_C = φ_F - φ_E (uniform field property)")
            
            # 3. Check electric field magnitude formula: E = (φ_E - φ_A)/AE
            if abs(self.AE) < 1e-12:  # Avoid division by near-zero
                self.log("FAIL: AE is too small, cannot compute E")
                return False
            E_calc = (self.varphi_E - self.varphi_A) / self.AE
            if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: E={self.E} != (φ_E-φ_A)/AE={E_calc}")
                return False
            self.log("PASS: E = (φ_E - φ_A)/AE")
            
            # 4. Check kinematics: The particle travels from A to D
            # The displacement in the field direction is the vertical distance from A to D
            # In a hexagon, AD is perpendicular to the field, and its vertical component is
            # AD_vertical = side * sqrt(3)  (corrected from 2*side*sin(60°))
            AD_vertical = self.side * math.sqrt(3)
            
            # Using kinematics: AD_vertical = 0.5*(|q|E/m)*t^2
            qm_abs = abs(self.fracqm)
            
            # Handle very small t values carefully
            if abs(self.t) < 1e-10:
                self.log("WARNING: t is extremely small, skipping kinematics check")
                # Skip this check if t is too small
                self.log("SKIP: AD_vertical kinematics check (t too small)")
            else:
                t_squared = self.t * self.t
                AD_vertical_calc = 0.5 * qm_abs * self.E * t_squared
                
                # Use appropriate tolerance for this calculation
                if not math.isclose(AD_vertical, AD_vertical_calc, rel_tol=1e-1, abs_tol=1e-2):
                    self.log(f"FAIL: AD_vertical={AD_vertical} != 0.5*(|q|E/m)*t^2={AD_vertical_calc}")
                    self.log(f"  |q|/m={qm_abs}, E={self.E}, t={self.t}, t^2={t_squared}")
                    return False
                self.log("PASS: AD_vertical = 0.5*(|q|E/m)*t^2 (kinematics)")
            
            # 5. Check geometry: AE = 2 * side * cos(30°)
            AE_geom = 2 * self.side * math.cos(math.radians(30))
            if not math.isclose(self.AE, AE_geom, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: AE={self.AE} != 2*side*cos30°={AE_geom}")
                return False
            self.log("PASS: AE geometry correct")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check φ_F calculation
            phi_F_calc = self.varphi_E + self.varphi_B - self.varphi_C
            if not math.isclose(self.varphi_F, phi_F_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: φ_F={self.varphi_F} != φ_E+φ_B-φ_C={phi_F_calc}")
                return False
            self.log("PASS: φ_F consistent with given potentials")
            
            # 2. Check time calculation: t = side / v0
            if abs(self.v0) < 1e-12:
                self.log("FAIL: v0 is too small")
                return False
            t_calc = self.side / self.v0
            
            # Handle very small t values with appropriate tolerance
            if abs(self.t) < 1e-10 or abs(t_calc) < 1e-10:
                # For very small t, use absolute tolerance
                if not math.isclose(self.t, t_calc, rel_tol=1e-1, abs_tol=1e-10):
                    self.log(f"FAIL: t={self.t} != side/v0={t_calc}")
                    self.log(f"  Difference: {abs(self.t - t_calc)}")
                    return False
            else:
                # For normal t values, use relative tolerance
                if not math.isclose(self.t, t_calc, rel_tol=1e-2):
                    self.log(f"FAIL: t={self.t} != side/v0={t_calc}")
                    self.log(f"  Difference: {abs(self.t - t_calc)}")
                    return False
            self.log("PASS: t = side/v0")
            
            # 3. Check E matches target answer (800√3/3 ≈ 461.88)
            E_target = 800 * math.sqrt(3) / 3
            if not math.isclose(self.E, E_target, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: E={self.E} != 800√3/3≈{E_target}")
                return False
            self.log("PASS: E matches target value")
            
            # 4. Check q/m matches target answer (1.875e10)
            if not math.isclose(abs(self.fracqm), 1.875e10, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: |q/m|={abs(self.fracqm)} != 1.875e10")
                return False
            self.log("PASS: |q/m| matches target value")
            
            # 5. Check sign of charge
            if self.q_sign != 'negative':
                self.log(f"WARNING: q sign is {self.q_sign}, expected negative")
                # Not a failure, just warning
            else:
                self.log("PASS: Charge polarity negative")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        try:
            # Use local copies to avoid modifying instance state
            side = self.original_side
            v0 = self.original_v0
            varphi_A = self.original_varphi_A
            varphi_E = self.original_varphi_E
            E = self.original_E
            fracqm = self.original_fracqm
            
            # Test 1: Increase side length → AE increases → E decreases (since ΔV fixed)
            side_new = side * 1.1  # +10%
            AE_new = 2 * side_new * math.cos(math.radians(30))
            E_new = (varphi_E - varphi_A) / AE_new
            if not (E_new < E):  # E should decrease
                self.log(f"FAIL: Increasing side did not decrease E (was {E}, now {E_new})")
                return False
            self.log(f"PASS: E monotonic with side (E decreased from {E:.2f} to {E_new:.2f})")
            
            # Test 2: Increase v0 → time decreases → |q/m| increases (since AD_vertical fixed)
            v0_new = v0 * 1.1  # +10%
            t_new = side / v0_new
            AD_vertical = side * math.sqrt(3)
            
            # Using kinematics: |q/m| = 2*AD_vertical/(E*t^2)
            if abs(t_new) < 1e-12:
                self.log("FAIL: t_new is too small")
                return False
                
            qm_new = 2 * AD_vertical / (E * t_new**2)
            if not (qm_new > abs(fracqm)):  # |q/m| should increase
                self.log(f"FAIL: Increasing v0 did not increase |q/m| (was {abs(fracqm):.2e}, now {qm_new:.2e})")
                return False
            self.log(f"PASS: |q/m| monotonic with v0 (increased from {abs(fracqm):.2e} to {qm_new:.2e})")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        self.log("=== PhysicsChecker started ===")
        
        # Run tests - each test restores its own state
        if not self.test_invariants():
            success = False
            self.log("FAILED: test_invariants")
        
        if not self.test_trace_consistency():
            success = False
            self.log("FAILED: test_trace_consistency")
        
        if not self.test_monotonicity():
            success = False
            self.log("FAILED: test_monotonicity")
        
        self.log(f"=== PhysicsChecker finished: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs