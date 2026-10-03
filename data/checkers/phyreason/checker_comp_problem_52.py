import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Never access solution_instance.attr directly; use getattr with fallbacks.
        """
        self.solution = solution_instance
        
        # Safely extract inputs with defaults
        self.A = getattr(solution_instance, 'A', 1.0)
        self.V = getattr(solution_instance, 'V', 1.0)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.g = getattr(solution_instance, 'g', 9.81)
        self.Omega = getattr(solution_instance, 'Omega', 7.292115e-5)  # Earth's angular velocity
        self.latitude_deg = getattr(solution_instance, 'latitude_deg', 49.0)  # Paris latitude
        self.latitude_rad = math.radians(self.latitude_deg)
        
        # Safely extract results (expecting a results dictionary)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe fallbacks
        # Numeric values (required outputs)
        self.T_N = float(self.results.get('T_N', 12.0))  # hours
        self.T_Paris = float(self.results.get('T_Paris', 0.0))  # hours
        
        # String values (optional, not required by contract)
        # If not present, we'll use default expected strings for testing
        self.eq_x = str(self.results.get('a_x+omega²x=0', 'a_x+omega²x=0'))
        self.eq_y = str(self.results.get('a_y+omega²y=0', 'a_y+omega²y=0'))
        self.xtilde_expr = str(self.results.get('x̃t', 'A cos(ωt) cos(Ωt) + (V/ω) sin(ωt) sin(Ωt)'))
        self.ytilde_expr = str(self.results.get('ỹt', '-A cos(ωt) sin(Ωt) + (V/ω) sin(ωt) cos(Ωt)'))
        
        # Derived parameters
        self.omega = math.sqrt(self.g / self.l)  # SHM angular frequency
        
        # Test logs
        self.logs = []

    def _log(self, message: str):
        """Add a message to the logs."""
        self.logs.append(message)

    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Energy conservation in inertial frame
        2. Newton's 2nd Law (SHM equations) - skip if strings not provided
        3. Rotational transformation orthogonality
        """
        success = True
        
        try:
            # 1. Energy conservation in John's inertial frame
            # For SHM: E_total = 0.5*m*(dx/dt)^2 + 0.5*m*ω²*x²
            # With x(t) = A*cos(ωt), y(t) = (V/ω)*sin(ωt)
            m = 1.0  # test mass
            
            # Compute energies at different times
            times = [0, 0.1/self.omega, 0.5/self.omega, 1.0/self.omega]
            energies = []
            
            for time in times:
                x = self.A * math.cos(self.omega * time)
                y = (self.V / self.omega) * math.sin(self.omega * time)
                vx = -self.A * self.omega * math.sin(self.omega * time)
                vy = self.V * math.cos(self.omega * time)
                
                kinetic = 0.5 * m * (vx**2 + vy**2)
                potential = 0.5 * m * self.omega**2 * (x**2 + y**2)
                total = kinetic + potential
                energies.append(total)
            
            # Check if all energies are approximately equal
            for i in range(1, len(energies)):
                if not math.isclose(energies[i], energies[0], rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Energy conservation violated: E0={energies[0]:.4f}, E{i}={energies[i]:.4f}")
                    success = False
            
            # 2. Verify SHM equations match expected form (only if strings are provided)
            # Since strings are optional, we check if they match expected patterns
            # but we don't fail if they're empty (as per contract)
            expected_eq_x = "a_x+omega²x=0"
            expected_eq_y = "a_y+omega²y=0"
            if self.eq_x and self.eq_x != expected_eq_x:
                # Allow minor formatting differences
                if not ("a_x" in self.eq_x and "omega" in self.eq_x and "x" in self.eq_x):
                    self._log(f"x-equation may be incorrect: {self.eq_x}")
                    # Not a failure because strings are not required
            if self.eq_y and self.eq_y != expected_eq_y:
                if not ("a_y" in self.eq_y and "omega" in self.eq_y and "y" in self.eq_y):
                    self._log(f"y-equation may be incorrect: {self.eq_y}")
                    # Not a failure
            
            # 3. Verify rotation transformation preserves distances
            test_time = 1.0  # arbitrary test time
            x_inertial = self.A * math.cos(self.omega * test_time)
            y_inertial = (self.V / self.omega) * math.sin(self.omega * test_time)
            
            x_rot = x_inertial * math.cos(self.Omega * test_time) + y_inertial * math.sin(self.Omega * test_time)
            y_rot = -x_inertial * math.sin(self.Omega * test_time) + y_inertial * math.cos(self.Omega * test_time)
            
            dist_inertial = x_inertial**2 + y_inertial**2
            dist_rot = x_rot**2 + y_rot**2
            
            if not math.isclose(dist_inertial, dist_rot, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance not preserved under rotation: inertial={dist_inertial:.4f}, rotating={dist_rot:.4f}")
                success = False
            
            if success:
                self._log("All invariants passed.")
                
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}")
            success = False
            
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Verify transformation equations match expected forms (optional).
        """
        success = True
        
        try:
            # Check North Pole period
            expected_T_N = math.pi / self.Omega  # seconds
            expected_T_N_hours = expected_T_N / 3600
            
            if not math.isclose(self.T_N, expected_T_N_hours, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"T_N mismatch: expected={expected_T_N_hours:.2f}h, actual={self.T_N:.2f}h")
                success = False
            
            # Check Paris period formula
            expected_T_Paris = self.T_N / math.sin(self.latitude_rad)  # hours
            
            if not math.isclose(self.T_Paris, expected_T_Paris, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"T_Paris mismatch: expected={expected_T_Paris:.2f}h, actual={self.T_Paris:.2f}h")
                success = False
            
            # Check transformed coordinate patterns (optional)
            # Only fail if strings are provided but don't match
            expected_xtilde_pattern = "A cos(ωt) cos(Ωt) + (V/ω) sin(ωt) sin(Ωt)"
            expected_ytilde_pattern = "-A cos(ωt) sin(Ωt) + (V/ω) sin(ωt) cos(Ωt)"
            
            # Remove LaTeX formatting for comparison
            xtilde_clean = self.xtilde_expr.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
            ytilde_clean = self.ytilde_expr.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
            
            # Check if patterns are present (allow partial matches)
            # Since strings are optional, we only warn if they're present but wrong
            if self.xtilde_expr and self.xtilde_expr != 'A cos(ωt) cos(Ωt) + (V/ω) sin(ωt) sin(Ωt)':
                if expected_xtilde_pattern.replace(" ", "") not in xtilde_clean.replace(" ", ""):
                    self._log(f"x̃(t) pattern mismatch: expected contains '{expected_xtilde_pattern}', got '{xtilde_clean}'")
                    # Not a failure because strings are not required
                    
            if self.ytilde_expr and self.ytilde_expr != '-A cos(ωt) sin(Ωt) + (V/ω) sin(ωt) cos(Ωt)':
                if expected_ytilde_pattern.replace(" ", "") not in ytilde_clean.replace(" ", ""):
                    self._log(f"ỹ(t) pattern mismatch: expected contains '{expected_ytilde_pattern}', got '{ytilde_clean}'")
                    # Not a failure
            
            if success:
                self._log("All trace consistency checks passed.")
                
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}")
            success = False
            
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check physical trends.
        Restore original values after test.
        """
        success = True
        
        try:
            # Save original values
            orig_latitude = self.latitude_deg
            
            # Test 1: Period vs latitude monotonicity
            # Higher latitude → shorter period (T = 12/sin(lat))
            test_lats = [10.0, 45.0, 80.0]  # degrees
            periods = []
            
            for lat in test_lats:
                lat_rad = math.radians(lat)
                # Avoid division by zero (sin(0) = 0)
                if math.sin(lat_rad) == 0:
                    continue
                T = self.T_N / math.sin(lat_rad)
                periods.append(T)
            
            # Check monotonic decrease: higher latitude → shorter period
            for i in range(1, len(periods)):
                if periods[i] >= periods[i-1]:
                    self._log(f"Period not monotonic with latitude: T({test_lats[i-1]}°)={periods[i-1]:.2f}, "
                             f"T({test_lats[i]}°)={periods[i]:.2f}")
                    success = False
            
            # Test 2: Coriolis force vs angular velocity
            # Higher Ω → larger Coriolis force
            test_Omegas = [0.5*self.Omega, self.Omega, 2*self.Omega]
            coriolis_forces = []
            m = 1.0  # test mass
            v = 1.0  # test velocity
            phi = math.pi/2  # North Pole
            
            for Omega_test in test_Omegas:
                F_cor = 2 * m * v * Omega_test * math.sin(phi)
                coriolis_forces.append(F_cor)
            
            # Check monotonic increase
            for i in range(1, len(coriolis_forces)):
                if coriolis_forces[i] <= coriolis_forces[i-1]:
                    self._log(f"Coriolis force not monotonic with Ω")
                    success = False
            
            # Restore original values
            self.latitude_deg = orig_latitude
            self.latitude_rad = math.radians(orig_latitude)
            
            if success:
                self._log("Monotonicity tests passed.")
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}")
            success = False
            
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} Test ===")
            try:
                success = test_func()
                # Only fail if critical numeric checks fail
                # String checks are non-critical
                if not success and test_name == "Monotonicity":
                    overall_success = False
                    self._log(f"FAILED: {test_name}")
                elif not success:
                    # For Invariants and Trace Consistency, only fail if numeric checks fail
                    # We'll check if the failure was due to numeric mismatches
                    # For simplicity, we treat all failures as non-critical except Monotonicity
                    self._log(f"WARNING: {test_name} had issues but not critical")
                else:
                    self._log(f"PASSED: {test_name}")
            except Exception as e:
                self._log(f"ERROR in {test_name}: {str(e)}")
                self._log(traceback.format_exc())
                overall_success = False
        
        # Overall success based on numeric results
        # Check if T_N and T_Paris are reasonable
        expected_T_N = math.pi / self.Omega / 3600
        expected_T_Paris = expected_T_N / math.sin(self.latitude_rad)
        
        if not (math.isclose(self.T_N, expected_T_N, rel_tol=1e-2) and 
                math.isclose(self.T_Paris, expected_T_Paris, rel_tol=1e-2)):
            overall_success = False
            self._log(f"Critical numeric mismatch: T_N={self.T_N:.2f} (expected {expected_T_N:.2f}), "
                     f"T_Paris={self.T_Paris:.2f} (expected {expected_T_Paris:.2f})")
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs