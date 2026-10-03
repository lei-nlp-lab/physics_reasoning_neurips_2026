import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access.
        solution_instance should have attributes or a results dictionary.
        """
        self.solution = solution_instance
        
        # Try to get results dictionary in multiple ways
        results = {}
        
        # Method 1: Check if solution_instance has a 'results' attribute
        if hasattr(solution_instance, 'results'):
            attr = getattr(solution_instance, 'results')
            if isinstance(attr, dict):
                results = attr
        
        # Method 2: Check if solution_instance itself is a dict
        elif isinstance(solution_instance, dict):
            results = solution_instance
        
        # Method 3: Check for common patterns
        else:
            # Look for any dict attribute that might contain results
            for attr_name in dir(solution_instance):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(solution_instance, attr_name)
                        if isinstance(attr, dict) and 'r' in attr and 'F_static' in attr:
                            results = attr
                            break
                    except:
                        pass
        
        self.results = results
        
        # Input parameters from problem statement - use defaults
        self.m = 2e-4  # kg
        self.l = 0.1   # m
        self.theta_deg = 37.0
        self.theta = math.radians(self.theta_deg)
        self.g = 10.0  # m/s²
        self.k = 9e9   # N·m²/C²
        
        # Extract values from results dictionary
        # Use direct access with None as default
        self.r = results.get('r')
        self.q_A = results.get('q_A')
        self.F_static = results.get('F_static')
        
        # For q_B, use the value from results if available, otherwise use problem value
        q_B_from_results = results.get('q_B')
        if q_B_from_results is not None:
            self.q_B = q_B_from_results
        else:
            self.q_B = 2e-7  # Default from problem statement
        
        # Expected values from solution steps (for comparison)
        self.r_expected = 0.06  # m
        self.F_static_expected = 1.5e-3  # N
        self.q_A_expected = 3e-9  # C
        
        self.logs = []
    
    def log(self, message: str):
        """Helper to append logs."""
        self.logs.append(message)
    
    def test_geometric_consistency(self) -> bool:
        """Check if r = l * sin(theta)."""
        success = True
        
        if self.r is None:
            self.log("ERROR: Distance r not found in results.")
            return False
        
        r_calculated = self.l * math.sin(self.theta)
        if not math.isclose(self.r, r_calculated, rel_tol=1e-2, abs_tol=1e-4):
            self.log(f"FAIL: Geometric inconsistency. r = {self.r}, expected l*sinθ = {r_calculated}")
            success = False
        else:
            self.log(f"PASS: r = {self.r:.4f} matches l*sinθ = {r_calculated:.4f}")
        
        return success
    
    def test_force_equilibrium(self) -> bool:
        """
        Verify force equilibrium:
        - Horizontal: T*sinθ = F_electrostatic
        - Vertical: T*cosθ = m*g
        => F_electrostatic = m*g*tanθ
        """
        success = True
        
        if self.F_static is None:
            self.log("ERROR: Electrostatic force F_static not found.")
            return False
        
        # Calculate expected force from equilibrium
        F_expected = self.m * self.g * math.tan(self.theta)
        
        if not math.isclose(self.F_static, F_expected, rel_tol=1e-2, abs_tol=1e-5):
            self.log(f"FAIL: Force equilibrium violated. F_static = {self.F_static}, expected m*g*tanθ = {F_expected}")
            success = False
        else:
            self.log(f"PASS: F_static = {self.F_static:.2e} matches m*g*tanθ = {F_expected:.2e}")
        
        # Also check net force is zero numerically
        T = self.m * self.g / math.cos(self.theta)  # Tension from vertical equilibrium
        F_horiz = T * math.sin(self.theta)
        if not math.isclose(F_horiz, self.F_static, rel_tol=1e-2, abs_tol=1e-5):
            self.log(f"FAIL: Horizontal force mismatch. T*sinθ = {F_horiz}, F_static = {self.F_static}")
            success = False
        
        return success
    
    def test_coulombs_law(self) -> bool:
        """
        Verify Coulomb's law: F = k * |q_A * q_B| / r^2
        Check both magnitude and sign (force should be repulsive for like charges).
        """
        success = True
        
        # Check for None values
        if None in (self.F_static, self.q_A, self.r, self.q_B):
            missing = []
            if self.F_static is None: missing.append("F_static")
            if self.q_A is None: missing.append("q_A")
            if self.r is None: missing.append("r")
            if self.q_B is None: missing.append("q_B")
            self.log(f"ERROR: Missing data for Coulomb's law: {missing}")
            return False
        
        # Check for zero values that would cause issues
        if self.r == 0:
            self.log("ERROR: Distance r is zero, cannot compute Coulomb's law.")
            return False
        
        if self.q_B == 0:
            self.log(f"ERROR: q_B is zero ({self.q_B}), cannot compute Coulomb's law.")
            return False
        
        # Calculate expected force from Coulomb's law
        try:
            F_coulomb = self.k * abs(self.q_A * self.q_B) / (self.r ** 2)
        except ZeroDivisionError:
            self.log("ERROR: Division by zero in Coulomb's law calculation.")
            return False
        except Exception as e:
            self.log(f"ERROR: Exception in Coulomb's law calculation: {e}")
            return False
        
        if not math.isclose(self.F_static, F_coulomb, rel_tol=1e-2, abs_tol=1e-5):
            self.log(f"FAIL: Coulomb's law mismatch. F_static = {self.F_static}, k*|q_A*q_B|/r² = {F_coulomb}")
            self.log(f"  q_A = {self.q_A:.2e}, q_B = {self.q_B:.2e}, r = {self.r:.4f}")
            success = False
        else:
            self.log(f"PASS: Coulomb's law satisfied. F_static = {self.F_static:.2e}, k|q_Aq_B|/r² = {F_coulomb:.2e}")
        
        # Check sign
        if self.q_A < 0:
            self.log("WARNING: q_A is negative, but geometry suggests repulsive force (like charges).")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        """
        success = True
        
        # Check r matches expected from solution steps
        if self.r is not None:
            if not math.isclose(self.r, self.r_expected, rel_tol=1e-2, abs_tol=1e-4):
                self.log(f"FAIL: Trace inconsistency in r. Got {self.r}, expected {self.r_expected}")
                success = False
            else:
                self.log(f"PASS: r = {self.r:.4f} matches expected {self.r_expected}")
        
        # Check F_static matches expected from solution steps
        if self.F_static is not None:
            if not math.isclose(self.F_static, self.F_static_expected, rel_tol=1e-2, abs_tol=1e-5):
                self.log(f"FAIL: Trace inconsistency in F_static. Got {self.F_static}, expected {self.F_static_expected}")
                success = False
            else:
                self.log(f"PASS: F_static = {self.F_static:.2e} matches expected {self.F_static_expected}")
        
        # Check q_A matches expected from solution steps
        if self.q_A is not None:
            if not math.isclose(self.q_A, self.q_A_expected, rel_tol=1e-2, abs_tol=1e-10):
                self.log(f"FAIL: Trace inconsistency in q_A. Got {self.q_A}, expected {self.q_A_expected}")
                success = False
            else:
                self.log(f"PASS: q_A = {self.q_A:.2e} matches expected {self.q_A_expected}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input (e.g., mass m) and verify the trend.
        Increase mass -> tension increases -> electrostatic force increases -> q_A increases.
        Then restore original values.
        """
        success = True
        
        if None in (self.F_static, self.q_A, self.r, self.q_B):
            self.log("SKIP: Monotonicity test due to missing data.")
            return True
        
        # Check for zero q_B to avoid division by zero
        if self.q_B == 0:
            self.log("SKIP: Monotonicity test - q_B is zero, would cause division by zero.")
            return True
        
        original_m = self.m
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        try:
            # New electrostatic force from equilibrium: F' = m' * g * tanθ
            F_static_new = perturbed_m * self.g * math.tan(self.theta)
            
            # New charge from Coulomb's law: q_A' = F' * r^2 / (k * q_B)
            q_A_new = F_static_new * (self.r ** 2) / (self.k * self.q_B)
            
            # Check monotonicity
            if not (F_static_new > self.F_static and q_A_new > self.q_A):
                self.log(f"FAIL: Monotonicity violated. Increasing m should increase both F and q_A.")
                self.log(f"  Original: m={original_m}, F={self.F_static:.2e}, q_A={self.q_A:.2e}")
                self.log(f"  Perturbed: m={perturbed_m}, F_new={F_static_new:.2e}, q_A_new={q_A_new:.2e}")
                success = False
            else:
                self.log(f"PASS: Monotonicity holds. Increasing m increases F_static and q_A.")
                
        except ZeroDivisionError:
            self.log("ERROR: Division by zero in monotonicity test.")
            success = False
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {e}")
            success = False
        
        return success
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants: net force zero (equilibrium).
        Use only numerical checks.
        """
        success = True
        
        if self.F_static is None:
            self.log("SKIP: Invariant test due to missing F_static.")
            return True
        
        # Numerical check: F_horizontal = m*g*tanθ should equal F_static
        F_horiz_expected = self.m * self.g * math.tan(self.theta)
        if not math.isclose(self.F_static, F_horiz_expected, rel_tol=1e-2, abs_tol=1e-5):
            self.log(f"FAIL: Numerical invariant broken. F_static = {self.F_static}, m*g*tanθ = {F_horiz_expected}")
            success = False
        else:
            self.log("PASS: Numerical equilibrium invariant holds.")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success flag and logs.
        """
        self.logs = []
        
        tests = [
            ("Geometric consistency", self.test_geometric_consistency),
            ("Force equilibrium", self.test_force_equilibrium),
            ("Coulomb's law", self.test_coulombs_law),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Invariants", self.test_invariants),
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"ERROR in {test_name}: {e}")
                overall_success = False
        
        self.log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs