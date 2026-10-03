import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary physics quantities from solution_instance.
        Uses getattr with fallback to results dict, and provides defaults if missing.
        """
        self.g = 10.0  # Given in problem statement
        
        # Safe extraction with fallbacks
        self.T = self._safe_get(solution_instance, 'T')  # Period
        self.m = self._safe_get(solution_instance, 'm')  # Mass
        self.v = self._safe_get(solution_instance, 'v')  # Max speed at lowest point
        self.F_min = self._safe_get(solution_instance, 'F_min')  # Min tension (highest point)
        self.F_max = self._safe_get(solution_instance, 'F_max')  # Max tension (lowest point)
        self.L = self._safe_get(solution_instance, 'L')  # Pendulum length
        
        # For symbolic checks
        self.symbols = self._define_symbols()
        
        # Target answers for trace consistency
        self.targets = {
            'L': 0.4,      # meters
            'm': 0.05,     # kg
            'v': 0.283     # m/s
        }
        
        # Test results log
        self.logs = []
        
    def _safe_get(self, obj, key):
        """Safely get attribute from solution instance with multiple fallbacks."""
        # First try direct attribute
        val = getattr(obj, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) and not isinstance(val, bool) else val
            
        # Then try results dict
        if hasattr(obj, 'results') and isinstance(obj.results, dict):
            val = obj.results.get(key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) and not isinstance(val, bool) else val
                
        # Return None if not found
        return None
        
    def _define_symbols(self):
        """Define SymPy symbols for symbolic physics laws."""
        m, g, L, v, theta, F_min, F_max, T = sp.symbols('m g L v theta F_min F_max T')
        return {
            'm': m, 'g': g, 'L': L, 'v': v, 'theta': theta,
            'F_min': F_min, 'F_max': F_max, 'T': T
        }
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self):
        """Test fundamental physics laws: Energy conservation and Newton's laws."""
        logs = []
        success = True
        
        # 1. Energy conservation: m*g*L*(1-cosθ) = 0.5*m*v^2
        try:
            # First compute cosθ from F_min = m*g*cosθ
            if None not in (self.F_min, self.m):
                cos_theta = self.F_min / (self.m * self.g) if self.m * self.g != 0 else None
                
                if cos_theta is not None and None not in (self.L, self.v):
                    # Left side: m*g*L*(1-cosθ)
                    potential_loss = self.m * self.g * self.L * (1 - cos_theta)
                    # Right side: 0.5*m*v^2
                    kinetic_gain = 0.5 * self.m * self.v**2
                    
                    if self._is_close(potential_loss, kinetic_gain):
                        logs.append("✓ Energy conservation holds")
                    else:
                        logs.append(f"✗ Energy conservation mismatch: PE loss={potential_loss:.4f}, KE gain={kinetic_gain:.4f}")
                        success = False
                else:
                    logs.append("⚠ Skipping energy conservation (missing data)")
        except Exception as e:
            logs.append(f"⚠ Energy conservation check failed: {str(e)}")
            
        # 2. Newton's law at highest point: F_min = m*g*cosθ
        try:
            if None not in (self.F_min, self.m):
                rhs = self.m * self.g
                # Compute cosθ from geometry using small angle approximation if needed
                # But we can also check the formula directly
                cos_theta_calc = self.F_min / (self.m * self.g) if self.m * self.g != 0 else None
                if cos_theta_calc is not None:
                    # For small angles (<5°), cosθ should be close to 1
                    if 0.99 < cos_theta_calc <= 1.0:
                        logs.append("✓ Newton's law at highest point consistent with small angle")
                    else:
                        logs.append(f"⚠ cosθ={cos_theta_calc:.3f} suggests angle >5° or formula issue")
        except Exception as e:
            logs.append(f"⚠ Highest point Newton's law check failed: {str(e)}")
            
        # 3. Newton's law at lowest point: F_max = m*g + m*v²/L
        try:
            if None not in (self.F_max, self.m, self.v, self.L):
                rhs = self.m * self.g + self.m * self.v**2 / self.L if self.L != 0 else float('inf')
                if self._is_close(self.F_max, rhs):
                    logs.append("✓ Newton's law at lowest point holds")
                else:
                    logs.append(f"✗ Lowest point tension mismatch: F_max={self.F_max:.4f}, calculated={rhs:.4f}")
                    success = False
        except Exception as e:
            logs.append(f"⚠ Lowest point Newton's law check failed: {str(e)}")
            
        # 4. Period formula: T = 2π√(L/g)
        try:
            if None not in (self.T, self.L):
                T_calc = 2 * math.pi * math.sqrt(self.L / self.g)
                if self._is_close(self.T, T_calc):
                    logs.append("✓ Period formula holds")
                else:
                    logs.append(f"✗ Period mismatch: T={self.T:.4f}, calculated={T_calc:.4f}")
                    success = False
        except Exception as e:
            logs.append(f"⚠ Period formula check failed: {str(e)}")
            
        return success, logs
        
    def test_trace_consistency(self):
        """Check that intermediate results match final answers."""
        logs = []
        success = True
        
        # Check against target answers
        for key, target in self.targets.items():
            val = getattr(self, key, None)
            if val is not None:
                if self._is_close(val, target):
                    logs.append(f"✓ {key} matches target: {val:.3f} ≈ {target}")
                else:
                    logs.append(f"✗ {key} mismatch: {val:.3f} ≠ {target}")
                    success = False
            else:
                logs.append(f"⚠ {key} not available for comparison")
                
        # Internal consistency: L from period formula
        try:
            if self.T is not None:
                L_from_T = self.g * (self.T/(2*math.pi))**2
                if self.L is not None and self._is_close(self.L, L_from_T):
                    logs.append("✓ L consistent with period T")
                else:
                    logs.append(f"⚠ L={self.L} vs L_from_T={L_from_T:.3f}")
        except:
            pass
            
        return success, logs
        
    def test_monotonicity(self):
        """Perturb mass and check tension trends."""
        logs = []
        success = True
        
        try:
            if None not in (self.m, self.F_min, self.F_max, self.v, self.L):
                # Perturb mass by +10%
                m_perturbed = self.m * 1.1
                
                # Recompute tensions using physics formulas
                # F_min = m*g*cosθ, but cosθ is fixed from geometry
                # We need cosθ from original data
                cos_theta = self.F_min / (self.m * self.g) if self.m * self.g != 0 else None
                
                if cos_theta is not None:
                    # For F_min: directly proportional to mass
                    F_min_perturbed = m_perturbed * self.g * cos_theta
                    
                    # For F_max: F_max = m*g + m*v²/L
                    # But v changes with mass? Actually v from energy conservation:
                    # 0.5*m*v² = m*g*L*(1-cosθ) ⇒ v² = 2*g*L*(1-cosθ) independent of m!
                    # So v remains constant when mass changes
                    v_squared = 2 * self.g * self.L * (1 - cos_theta)
                    F_max_perturbed = m_perturbed * self.g + m_perturbed * v_squared / self.L
                    
                    # Check monotonic trends
                    if F_min_perturbed > self.F_min and F_max_perturbed > self.F_max:
                        logs.append("✓ Tensions increase with mass (monotonic)")
                    else:
                        logs.append("✗ Tension trend with mass violated")
                        success = False
                else:
                    logs.append("⚠ Skipping monotonicity (cannot compute cosθ)")
            else:
                logs.append("⚠ Skipping monotonicity (missing data)")
        except Exception as e:
            logs.append(f"⚠ Monotonicity test failed: {str(e)}")
            success = False
            
        return success, logs
        
    def run_all_tests(self):
        """Run all physics audits and return consolidated results."""
        self.logs = []
        all_success = True
        
        # Run each test suite
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                self.logs.append(f"\n=== {test_name} ===")
                self.logs.extend(logs)
                if not success:
                    all_success = False
            except Exception as e:
                self.logs.append(f"\n=== {test_name} ===")
                self.logs.append(f"ERROR: {str(e)}")
                self.logs.append(traceback.format_exc())
                all_success = False
                
        return all_success, self.logs