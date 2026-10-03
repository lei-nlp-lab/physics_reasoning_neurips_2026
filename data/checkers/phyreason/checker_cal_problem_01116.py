import math
import traceback
from typing import List, Tuple, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Ball motion parameters (from problem statement)
        self.v0 = 10.0  # m/s
        self.g = 10.0   # m/s²
        self.v_ground = -20.0  # m/s (negative since downward)
        self.L = 5.0    # m
        self.s = 4.0    # m
        
        # Extract computed results with safe access
        self.t = self._safe_extract('t', 0.0)
        self.h = self._safe_extract('h', 0.0)
        self.H = self._safe_extract('H', 0.0)
        
        # Extract acceleration values - check for both individual and combined formats
        self.a_min = self._safe_extract('a_min', None)
        self.a_max = self._safe_extract('a_max', None)
        
        # If individual values not found, try to extract from 'a' field
        if self.a_min is None or self.a_max is None:
            a_val = self.results.get('a', None)
            if isinstance(a_val, (tuple, list)) and len(a_val) >= 2:
                self.a_min = float(a_val[0])
                self.a_max = float(a_val[1])
            elif isinstance(a_val, dict):
                self.a_min = a_val.get('min', 0.0)
                self.a_max = a_val.get('max', 0.0)
            elif a_val is not None:
                # Single value case - use it for both bounds
                try:
                    a_float = float(a_val)
                    self.a_min = a_float
                    self.a_max = a_float
                except:
                    pass
    
    def _safe_extract(self, key: str, default: Any) -> Any:
        """Safely extract a value from results."""
        val = self.results.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except:
            return val
    
    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        # 1. Energy conservation for ball (per unit mass)
        # Initial energy: KE0 + PE0 = 0.5*v0² + g*H
        # Final energy: KE_ground + PE_ground = 0.5*v_ground² + 0
        E_initial = 0.5*self.v0**2 + self.g*self.H
        E_final = 0.5*self.v_ground**2
        if not math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy conservation failed: E_initial={E_initial:.3f}, E_final={E_final:.3f}")
        else:
            logs.append("✓ Energy conservation satisfied")
        
        # 2. Kinematic equation for ball: v_ground = v0 - g*t
        v_calc = self.v0 - self.g*self.t
        if not math.isclose(v_calc, self.v_ground, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Kinematic eq failed: v0 - g*t = {v_calc:.3f}, expected {self.v_ground:.3f}")
        else:
            logs.append("✓ Ball kinematics satisfied")
        
        # 3. Displacement relation: h = v0*t - 0.5*g*t²
        h_calc = self.v0*self.t - 0.5*self.g*self.t**2
        if not math.isclose(h_calc, self.h, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Displacement eq failed: h_calc={h_calc:.3f}, h={self.h:.3f}")
        else:
            logs.append("✓ Ball displacement consistent")
        
        # 4. Height relation: H = |h|
        if not math.isclose(abs(self.h), self.H, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Height relation failed: |h|={abs(self.h):.3f}, H={self.H:.3f}")
        else:
            logs.append("✓ Height relation satisfied")
        
        # 5. Cart acceleration bounds logic
        if self.a_min is not None and self.a_max is not None:
            # Minimum acceleration: front of cart reaches landing point
            a_min_calc = 2*self.s / (self.t**2)
            if not math.isclose(self.a_min, a_min_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"a_min mismatch: {self.a_min:.3f} vs calculated {a_min_calc:.3f}")
            else:
                logs.append("✓ Minimum acceleration correct")
            
            # Maximum acceleration: back of cart reaches landing point
            a_max_calc = 2*(self.s + self.L) / (self.t**2)
            if not math.isclose(self.a_max, a_max_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"a_max mismatch: {self.a_max:.3f} vs calculated {a_max_calc:.3f}")
            else:
                logs.append("✓ Maximum acceleration correct")
            
            # Logical consistency: a_min ≤ a_max
            if self.a_min > self.a_max + 1e-2:  # Small tolerance for floating point
                logs.append(f"Logical error: a_min={self.a_min:.3f} > a_max={self.a_max:.3f}")
            else:
                logs.append("✓ Acceleration bounds logically consistent")
        else:
            logs.append("⚠ Acceleration bounds not available for testing")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # Check if t=3s leads to correct height
        if math.isclose(self.t, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            h_expected = self.v0*3 - 0.5*self.g*9
            H_expected = abs(h_expected)
            if (math.isclose(self.h, h_expected, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.H, H_expected, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append("✓ Time t=3s consistent with given ball data")
            else:
                logs.append(f"✗ With t=3s, expected h={h_expected:.2f}, H={H_expected:.2f}")
        else:
            logs.append(f"Note: t={self.t:.3f} differs from expected 3.0s")
        
        # Check if using t yields correct acceleration bounds
        if self.a_min is not None and self.a_max is not None:
            a_min_from_t = 2*self.s / (self.t**2)
            a_max_from_t = 2*(self.s + self.L) / (self.t**2)
            if (math.isclose(self.a_min, a_min_from_t, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.a_max, a_max_from_t, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append("✓ Acceleration bounds consistent with computed t")
            else:
                logs.append(f"✗ With t={self.t:.3f}, expected a_min={a_min_from_t:.3f}, a_max={a_max_from_t:.3f}")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test how results change when perturbing inputs."""
        logs = []
        
        # Perturb initial velocity by +10%
        v0_perturbed = self.v0 * 1.10
        
        # Recompute time using kinematic equation: v_ground = v0 - g*t
        t_perturbed = (v0_perturbed - self.v_ground) / self.g  # v_ground is negative
        
        # Time should increase with increased initial velocity
        if t_perturbed > self.t:
            logs.append("✓ Monotonicity: t increases with increased v0")
        else:
            logs.append(f"✗ Monotonicity violated: t_perturbed={t_perturbed:.3f} ≤ t={self.t:.3f}")
        
        # Perturb horizontal distance s by +10%
        s_perturbed = self.s * 1.10
        
        # Recompute acceleration bounds using t (assuming t remains valid approximation)
        a_min_perturbed = 2*s_perturbed / (self.t**2)
        a_max_perturbed = 2*(s_perturbed + self.L) / (self.t**2)
        
        if self.a_min is not None and self.a_max is not None:
            if (a_min_perturbed > self.a_min and a_max_perturbed > self.a_max):
                logs.append("✓ Monotonicity: acceleration bounds increase with increased s")
            else:
                logs.append(f"✗ Monotonicity violated for s perturbation")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        logs = []
        
        # Run individual test suites
        logs.append("=== Testing Physics Invariants ===")
        logs.extend(self.test_invariants())
        
        logs.append("\n=== Testing Trace Consistency ===")
        logs.extend(self.test_trace_consistency())
        
        logs.append("\n=== Testing Monotonicity ===")
        logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        error_keywords = ['failed', 'mismatch', 'error', 'violated', '✗']
        for log in logs:
            if any(keyword in log.lower() for keyword in error_keywords):
                success = False
                break
        
        return success, logs