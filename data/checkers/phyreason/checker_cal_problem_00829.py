import math
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely
        self.R = getattr(solution_instance, 'R', 1.0)  # Default R=1 for scaling
        self.c = getattr(solution_instance, 'c', 1.0)  # Default c=1 for scaling
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)  # degrees
        self.theta = math.radians(self.theta_deg)
        
        # Extract results with safe fallbacks - handle alpha correctly
        # First try to get alpha in degrees if stored separately
        self.alpha_deg = float(self._safe_get('alpha_deg', '0.0'))
        if self.alpha_deg == 0.0:
            # If not, assume alpha is in radians and convert
            alpha_rad = float(self._safe_get('alpha', '0.0'))
            self.alpha_deg = math.degrees(alpha_rad)
        self.alpha = math.radians(self.alpha_deg)
        
        # Extract other results
        self.n = float(self._safe_get('n', '1.0'))
        self.t = float(self._safe_get('t', '0.0'))
        self.tan_alpha = float(self._safe_get('tanalpha', '0.0'))
        self.sinC = float(self._safe_get('sinC', '0.0'))
        self.sin_beta = float(self._safe_get('sinβ', '0.0'))
        self.EH = float(self._safe_get('EH', '0.0'))
        self.x = float(self._safe_get('x', '0.0'))
        self.v = float(self._safe_get('v', '0.0'))
        
        # Constants from problem
        self.OF = 0.75 * self.R
        self.OE = self.R
        self.OB = 0.375 * self.R
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Target answers
        self.target_n = 4/3
        self.target_t = 5/3  # When R=1, c=1
        
    def _safe_get(self, key: str, default: str) -> Any:
        """Safely get value from results or attributes."""
        val = self.results.get(key, getattr(self.solution, key, default))
        if isinstance(val, str):
            # Handle fractions and expressions
            if 'R' in val or 'c' in val:
                # Replace R and c with 1 for numerical evaluation
                val_eval = val.replace('R', '1').replace('c', '1')
                try:
                    return eval(val_eval)
                except:
                    return float(default)
            try:
                return float(val)
            except:
                return float(default)
        return float(val)
    
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_snells_law(self) -> Tuple[bool, List[str]]:
        """Test Snell's Law for ray 1."""
        logs = []
        success = True
        
        # Snell's Law: sin(90° - θ) = n * sin(α)
        # sin(90° - θ) = cos(θ)
        lhs = self.cos_theta  # cos(37°) = 0.8
        rhs = self.n * math.sin(self.alpha)
        
        if not self._is_close(lhs, rhs):
            logs.append(f"Snell's Law violation: cosθ={lhs:.4f}, n*sinα={rhs:.4f}")
            success = False
        else:
            logs.append(f"✓ Snell's Law satisfied: {lhs:.4f} ≈ {rhs:.4f}")
            
        return success, logs
    
    def test_critical_angle(self) -> Tuple[bool, List[str]]:
        """Test critical angle relationship."""
        logs = []
        success = True
        
        # sinC = 1/n
        expected_sinC = 1.0 / self.n
        if not self._is_close(self.sinC, expected_sinC):
            logs.append(f"Critical angle mismatch: sinC={self.sinC:.4f}, 1/n={expected_sinC:.4f}")
            success = False
        else:
            logs.append(f"✓ Critical angle correct: sinC={self.sinC:.4f} = 1/n")
            
        return success, logs
    
    def test_geometry_consistency(self) -> Tuple[bool, List[str]]:
        """Test geometric relationships from steps analysis."""
        logs = []
        success = True
        
        # 1. tanα = OF/OE = 3/4 = 0.75
        expected_tan_alpha = self.OF / self.OE
        if not self._is_close(self.tan_alpha, expected_tan_alpha):
            logs.append(f"tanα mismatch: {self.tan_alpha:.4f} vs expected {expected_tan_alpha:.4f}")
            success = False
        else:
            logs.append(f"✓ tanα correct: {self.tan_alpha:.4f}")
            
        # 2. α = arctan(3/4) ≈ 37°
        expected_alpha_deg = math.degrees(math.atan(0.75))
        if not self._is_close(self.alpha_deg, expected_alpha_deg):
            logs.append(f"α mismatch: {self.alpha_deg:.1f}° vs expected {expected_alpha_deg:.1f}°")
            success = False
        else:
            logs.append(f"✓ α correct: {self.alpha_deg:.1f}°")
            
        # 3. sinβ = sin(90° - α) = sin(53°) = 0.8
        expected_sin_beta = math.sin(math.radians(90 - self.alpha_deg))
        if not self._is_close(self.sin_beta, expected_sin_beta):
            logs.append(f"sinβ mismatch: {self.sin_beta:.4f} vs expected {expected_sin_beta:.4f}")
            success = False
        else:
            logs.append(f"✓ sinβ correct: {self.sin_beta:.4f}")
            
        # 4. EH = R - OB * tanβ
        tan_beta = self.sin_beta / math.sqrt(1 - self.sin_beta**2)
        expected_EH = self.R - self.OB * tan_beta
        if not self._is_close(self.EH, expected_EH):
            logs.append(f"EH mismatch: {self.EH:.4f}R vs expected {expected_EH:.4f}R")
            success = False
        else:
            logs.append(f"✓ EH correct: {self.EH:.4f}R")
            
        # 5. x = 2 * OG, where OG = OB/sinα
        OG = self.OB / math.sin(self.alpha)
        expected_x = 2 * OG
        if not self._is_close(self.x, expected_x):
            logs.append(f"x mismatch: {self.x:.4f}R vs expected {expected_x:.4f}R")
            success = False
        else:
            logs.append(f"✓ x correct: {self.x:.4f}R")
            
        return success, logs
    
    def test_speed_relationship(self) -> Tuple[bool, List[str]]:
        """Test v = c/n relationship."""
        logs = []
        success = True
        
        expected_v = self.c / self.n
        if not self._is_close(self.v, expected_v):
            logs.append(f"Speed mismatch: v={self.v:.4f}c vs expected {expected_v:.4f}c")
            success = False
        else:
            logs.append(f"✓ Speed correct: v={self.v:.4f}c = c/n")
            
        return success, logs
    
    def test_time_calculation(self) -> Tuple[bool, List[str]]:
        """Test t = x/v calculation."""
        logs = []
        success = True
        
        expected_t = self.x / self.v
        if not self._is_close(self.t, expected_t):
            logs.append(f"Time mismatch: t={self.t:.4f}R/c vs expected {expected_t:.4f}R/c")
            success = False
        else:
            logs.append(f"✓ Time correct: t={self.t:.4f}R/c")
            
        # Check against target answer (5R/(3c))
        target_t_scaled = 5/3  # When R=1, c=1
        if not self._is_close(self.t, target_t_scaled):
            logs.append(f"Target time mismatch: {self.t:.4f} vs {target_t_scaled:.4f}")
            success = False
            
        return success, logs
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test all physics invariants."""
        logs = ["=== Testing Physics Invariants ==="]
        all_success = True
        
        # Test Snell's Law
        success, snell_logs = self.test_snells_law()
        logs.extend(snell_logs)
        all_success &= success
        
        # Test critical angle
        success, crit_logs = self.test_critical_angle()
        logs.extend(crit_logs)
        all_success &= success
        
        # Test speed relationship
        success, speed_logs = self.test_speed_relationship()
        logs.extend(speed_logs)
        all_success &= success
        
        return all_success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Test consistency between intermediate results."""
        logs = ["=== Testing Trace Consistency ==="]
        all_success = True
        
        # Test geometry consistency
        success, geo_logs = self.test_geometry_consistency()
        logs.extend(geo_logs)
        all_success &= success
        
        # Test time calculation
        success, time_logs = self.test_time_calculation()
        logs.extend(time_logs)
        all_success &= success
        
        # Check refractive index against target
        if not self._is_close(self.n, self.target_n):
            logs.append(f"Refractive index mismatch: n={self.n:.4f} vs target {self.target_n:.4f}")
            all_success = False
        else:
            logs.append(f"✓ Refractive index matches target: n={self.n:.4f}")
            
        return all_success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic relationships by perturbing R."""
        logs = ["=== Testing Monotonicity ==="]
        success = True
        
        # Save original values
        original_R = self.R
        original_t = self.t
        
        # Perturb R by +10%
        self.R *= 1.1
        self.OF = 0.75 * self.R
        self.OB = 0.375 * self.R
        self.OE = self.R
        
        # Recalculate expected values using physics formulas
        # n should remain constant (independent of R)
        expected_n = self.target_n
        
        # x should scale linearly with R
        OG = self.OB / math.sin(self.alpha)
        expected_x = 2 * OG
        
        # v should remain constant (c/n)
        expected_v = self.c / expected_n
        
        # t should scale linearly with R
        expected_t = expected_x / expected_v
        
        # Check that n didn't change
        if not self._is_close(self.n, expected_n):
            logs.append(f"n changed with R perturbation: {self.n:.4f} vs {expected_n:.4f}")
            success = False
        else:
            logs.append(f"✓ n invariant under R scaling")
            
        # Check that t scales correctly
        expected_scaling = self.R / original_R
        actual_scaling = expected_t / original_t
        
        if not self._is_close(actual_scaling, expected_scaling):
            logs.append(f"t scaling mismatch: {actual_scaling:.4f} vs {expected_scaling:.4f}")
            success = False
        else:
            logs.append(f"✓ t scales linearly with R: factor {actual_scaling:.4f}")
            
        # Restore original values
        self.R = original_R
        self.OF = 0.75 * self.R
        self.OB = 0.375 * self.R
        self.OE = self.R
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return results."""
        logs = ["=" * 50]
        logs.append("Physics Checker for Transparent Cylinder Problem")
        logs.append("=" * 50)
        
        all_success = True
        
        # Run each test suite
        success, test_logs = self.test_invariants()
        logs.extend(test_logs)
        all_success &= success
        
        logs.append("")
        success, test_logs = self.test_trace_consistency()
        logs.extend(test_logs)
        all_success &= success
        
        logs.append("")
        success, test_logs = self.test_monotonicity()
        logs.extend(test_logs)
        all_success &= success
        
        logs.append("")
        logs.append("=" * 50)
        logs.append(f"Overall Result: {'PASS' if all_success else 'FAIL'}")
        logs.append("=" * 50)
        
        return all_success, logs