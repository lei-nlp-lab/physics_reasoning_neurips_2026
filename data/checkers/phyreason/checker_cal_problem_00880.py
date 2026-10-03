import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # DEBUG: Print what we're getting
        # print(f"DEBUG: Results from solution: {self.results}")
        
        # Extract all needed values from results
        # Use EXACTLY the values from results, no defaults
        self.E_k = self.results.get('E_k')
        self.E_km = self.results.get('E_km')
        self.T = self.results.get('T')
        self.f = self.results.get('f')
        self.e = self.results.get('e')
        self.m = self.results.get('m')
        
        # The code uses these hardcoded values
        self.U = 1000.0  # V
        self.B = 0.5     # T
        self.R = 0.5     # m
        
        # Check if we got all required values
        self.missing_values = []
        for name, value in [('E_k', self.E_k), ('E_km', self.E_km), ('T', self.T), 
                           ('f', self.f), ('e', self.e), ('m', self.m)]:
            if value is None:
                self.missing_values.append(name)

    def _isclose(self, a, b):
        # Handle None values
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        logs = []
        try:
            # Check if we have all required values
            if self.missing_values:
                logs.append(f"ERROR: Missing values in results: {self.missing_values}")
                return logs
            
            # Check for zero values that would cause division errors
            if self.m == 0:
                logs.append("ERROR: Mass is zero in results")
                return logs
            if self.e == 0:
                logs.append("ERROR: Charge is zero in results")
                return logs
            if self.B == 0:
                logs.append("ERROR: Magnetic field is zero")
                return logs
            
            # 1. Initial kinetic energy from voltage acceleration
            expected_E_k = self.e * self.U
            if not self._isclose(self.E_k, expected_E_k):
                logs.append(f"FAIL: Initial kinetic energy mismatch. Got {self.E_k}, expected {expected_E_k}")
            else:
                logs.append("PASS: E_k = eU (Work-energy theorem)")

            # 2. Maximum kinetic energy from cyclotron
            expected_E_km = (self.e**2 * self.B**2 * self.R**2) / (2 * self.m)
            if not self._isclose(self.E_km, expected_E_km):
                logs.append(f"FAIL: Maximum kinetic energy mismatch. Got {self.E_km}, expected {expected_E_km}")
            else:
                logs.append("PASS: E_km = (e²B²R²)/(2m) (from circular motion & kinetic energy)")

            # 3. Period from circular motion
            expected_T = (2 * math.pi * self.m) / (self.e * self.B)
            if not self._isclose(self.T, expected_T):
                logs.append(f"FAIL: Period mismatch. Got {self.T}, expected {expected_T}")
            else:
                logs.append("PASS: T = 2πm/(eB) (period of circular motion in B field)")

            # 4. Frequency as reciprocal of period
            expected_f = 1 / self.T
            if not self._isclose(self.f, expected_f):
                logs.append(f"FAIL: Frequency not reciprocal of period. Got f={self.f}, 1/T={expected_f}")
            else:
                logs.append("PASS: f = 1/T (frequency-period relationship)")

            # 5. Direct frequency formula
            expected_f_direct = (self.e * self.B) / (2 * math.pi * self.m)
            if not self._isclose(self.f, expected_f_direct):
                logs.append(f"FAIL: Direct frequency mismatch. Got {self.f}, expected {expected_f_direct}")
            else:
                logs.append("PASS: f = eB/(2πm) (from period formula)")

            # 6. Newton's second law for circular motion (numeric check)
            expected_v = (self.e * self.B * self.R) / self.m
            E_km_from_v = 0.5 * self.m * expected_v**2
            if not self._isclose(self.E_km, E_km_from_v):
                logs.append(f"FAIL: Circular motion velocity inconsistent. From v: E_km={E_km_from_v}, stored={self.E_km}")
            else:
                logs.append("PASS: Circular motion (evB = mv²/R) consistent with E_km")

        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in test_invariants: {e}")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check if we have all required values
            if self.missing_values:
                logs.append(f"ERROR: Missing values in results: {self.missing_values}")
                return logs
            
            # Check for zero values
            if self.m == 0 or self.e == 0 or self.B == 0:
                logs.append("ERROR: Zero value in mass, charge, or magnetic field")
                return logs
            
            # Step 1: E_k = eU (already checked in invariants)
            expected_E_k = self.e * self.U
            if not self._isclose(self.E_k, expected_E_k):
                logs.append(f"FAIL: Step 1 (E_k = eU) inconsistent. Got {self.E_k}, expected {expected_E_k}")

            # Step 2 & 3: Circular motion leads to v = eBR/m, then E_km = 0.5*m*v²
            v_calc = (self.e * self.B * self.R) / self.m
            E_km_calc = 0.5 * self.m * v_calc**2
            if not self._isclose(self.E_km, E_km_calc):
                logs.append(f"FAIL: Step 2-3 inconsistent. From v={v_calc}, E_km={E_km_calc}, stored={self.E_km}")
            else:
                logs.append("PASS: Steps 2-3 consistent (circular motion → kinetic energy)")

            # Step 5 & 6: T = 2πm/(eB) and f = 1/T
            T_calc = (2 * math.pi * self.m) / (self.e * self.B)
            f_calc = 1 / T_calc
            if not (self._isclose(self.T, T_calc) and self._isclose(self.f, f_calc)):
                logs.append(f"FAIL: Steps 5-6 inconsistent. T={T_calc}, f={f_calc}, stored T={self.T}, f={self.f}")
            else:
                logs.append("PASS: Steps 5-6 consistent (T and f formulas)")

            # Cross-check: initial and maximum kinetic energies should satisfy E_km >= E_k
            if self.E_km < self.E_k and not self._isclose(self.E_km, self.E_k):
                logs.append(f"WARN: E_km ({self.E_km}) < E_k ({self.E_k}) - cyclotron should accelerate further")
            else:
                logs.append("PASS: E_km >= E_k (cyclotron acceleration logic)")

        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in test_trace_consistency: {e}")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Check if we have all required values
            if self.missing_values:
                logs.append(f"ERROR: Missing values in results: {self.missing_values}")
                return logs
            
            # Check for zero values
            if self.m == 0 or self.e == 0 or self.B == 0:
                logs.append("ERROR: Zero value in mass, charge, or magnetic field")
                return logs
            
            # Perturb mass by +1%
            perturb_factor = 1.01
            perturbed_m = self.m * perturb_factor

            # Recalculate expected results using formulas
            # 1. E_k should be independent of m
            original_E_k = self.E_k
            perturbed_E_k = self.e * self.U  # same as original
            if not self._isclose(original_E_k, perturbed_E_k):
                logs.append("FAIL: E_k changed with mass perturbation (should be independent)")
            else:
                logs.append("PASS: E_k independent of mass (only depends on U)")

            # 2. E_km should decrease with increased m (inversely proportional)
            original_E_km = self.E_km
            perturbed_E_km = (self.e**2 * self.B**2 * self.R**2) / (2 * perturbed_m)
            if original_E_km <= perturbed_E_km and not self._isclose(original_E_km, perturbed_E_km):
                logs.append(f"FAIL: E_km monotonicity violated. Original {original_E_km}, perturbed {perturbed_E_km}")
            else:
                logs.append("PASS: E_km decreases with increased mass (inverse proportionality)")

            # 3. T should increase with increased m (proportional)
            original_T = self.T
            perturbed_T = (2 * math.pi * perturbed_m) / (self.e * self.B)
            if original_T >= perturbed_T and not self._isclose(original_T, perturbed_T):
                logs.append(f"FAIL: T monotonicity violated. Original {original_T}, perturbed {perturbed_T}")
            else:
                logs.append("PASS: T increases with increased mass (proportional)")

            # 4. f should decrease with increased m (inversely proportional)
            original_f = self.f
            perturbed_f = (self.e * self.B) / (2 * math.pi * perturbed_m)
            if original_f <= perturbed_f and not self._isclose(original_f, perturbed_f):
                logs.append(f"FAIL: f monotonicity violated. Original {original_f}, perturbed {perturbed_f}")
            else:
                logs.append("PASS: f decreases with increased mass (inverse proportionality)")

        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in test_monotonicity: {e}")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        all_logs = []
        success = True

        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            logs = test_func()
            all_logs.extend(logs)
            # Check for any FAIL or ERROR in logs
            for log in logs:
                if log.startswith("FAIL") or log.startswith("ERROR"):
                    success = False

        return success, all_logs