import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance=None):
        """
        Extracts inputs and results from the global namespace or solution_instance.
        Accepts solution_instance parameter for compatibility with sandbox interface.
        """
        # Try to get values from global namespace first
        import __main__ as main
        
        # Inputs from problem context
        self.M = getattr(main, 'M', None)
        self.G = getattr(main, 'G', None)
        self.R = getattr(main, 'R', None)
        self.t = getattr(main, 't', None)
        self.n = getattr(main, 'n', None)
        
        # Results - try direct attributes first, then results dict
        self.T = getattr(main, 'T', None)
        self.h = getattr(main, 'h', None)
        self.v = getattr(main, 'v', None)
        
        # If not found as attributes, try results dictionary
        if self.T is None or self.h is None or self.v is None:
            results = getattr(main, 'results', {})
            if self.T is None:
                self.T = results.get('T', None)
            if self.h is None:
                self.h = results.get('h', None)
            if self.v is None:
                self.v = results.get('v', None)
        
        # If still None, try to get from solution_instance if provided
        if solution_instance is not None:
            if self.G is None:
                self.G = getattr(solution_instance, 'G', None)
            if self.M is None:
                self.M = getattr(solution_instance, 'M', None)
            if self.R is None:
                self.R = getattr(solution_instance, 'R', None)
            if self.t is None:
                self.t = getattr(solution_instance, 't', None)
            if self.n is None:
                self.n = getattr(solution_instance, 'n', None)
            if self.T is None:
                self.T = getattr(solution_instance, 'T', None)
            if self.h is None:
                self.h = getattr(solution_instance, 'h', None)
            if self.v is None:
                self.v = getattr(solution_instance, 'v', None)
            
            # Also try results dict from solution_instance
            if any([self.T is None, self.h is None, self.v is None]):
                results_dict = getattr(solution_instance, 'results', {})
                if self.T is None:
                    self.T = results_dict.get('T', None)
                if self.h is None:
                    self.h = results_dict.get('h', None)
                if self.v is None:
                    self.v = results_dict.get('v', None)

        # Logging
        self.logs = []

    def _compare(self, val1, val2, name=""):
        """
        Robust comparison: uses math.isclose for numerics.
        Returns True if values match within tolerance.
        """
        try:
            # If both are numeric (or convertible to float)
            num1 = float(val1)
            num2 = float(val2)
            # Use lenient tolerance for physics data
            if math.isclose(num1, num2, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"✓ {name}: {num1} ≈ {num2}")
                return True
            else:
                self.logs.append(f"✗ {name}: {num1} != {num2} (outside tolerance)")
                return False
        except (TypeError, ValueError):
            self.logs.append(f"⚠ {name}: Cannot compare non-numeric values")
            return False

    def test_invariants(self):
        """
        Checks fundamental physics laws:
        1. Period definition T = t/n
        2. Force balance: Gravitational force = Centripetal force
        3. Energy conservation (virial theorem for circular orbit)
        """
        success = True

        # 1. Period definition
        if self.T is not None and self.t is not None and self.n is not None:
            T_calc = self.t / self.n
            if not self._compare(self.T, T_calc, "Period T = t/n"):
                success = False
        else:
            self.logs.append(f"⚠ Missing data for period check: T={self.T}, t={self.t}, n={self.n}")
            success = False

        # 2. Force balance: G*M/(R+h)^2 = (2π/T)^2 * (R+h)
        # FIX: Check if any value is 0.0 which might cause division by zero
        if all([self.G is not None, self.M is not None, self.R is not None, 
                self.h is not None, self.T is not None]):
            try:
                # Additional check for zero values that would cause division errors
                if self.T == 0:
                    self.logs.append("⚠ Force balance: T is zero, cannot compute")
                    success = False
                elif (self.R + self.h) == 0:
                    self.logs.append("⚠ Force balance: R+h is zero, cannot compute")
                    success = False
                else:
                    # Numerical check
                    lhs = self.G * self.M / (self.R + self.h)**2
                    rhs = (2 * math.pi / self.T)**2 * (self.R + self.h)
                    if not self._compare(lhs, rhs, "Force balance (grav = centripetal)"):
                        success = False
            except ZeroDivisionError:
                self.logs.append("⚠ Force balance: Division by zero")
                success = False
            except Exception as e:
                self.logs.append(f"⚠ Force balance numerical check failed: {e}")
                success = False
        else:
            missing = []
            if self.G is None: missing.append('G')
            if self.M is None: missing.append('M')
            if self.R is None: missing.append('R')
            if self.h is None: missing.append('h')
            if self.T is None: missing.append('T')
            self.logs.append(f"⚠ Missing data for force balance: {missing}")
            success = False

        # 3. Virial theorem: v^2 = G*M/(R+h)
        if all([self.v is not None, self.G is not None, self.M is not None, 
                self.R is not None, self.h is not None]):
            try:
                if (self.R + self.h) == 0:
                    self.logs.append("⚠ Virial theorem: R+h is zero, cannot compute")
                    success = False
                else:
                    expected_v2 = self.G * self.M / (self.R + self.h)
                    if not self._compare(self.v**2, expected_v2, "Virial theorem v² = GM/(R+h)"):
                        success = False
            except ZeroDivisionError:
                self.logs.append("⚠ Virial theorem: Division by zero")
                success = False
            except Exception as e:
                self.logs.append(f"⚠ Virial theorem check failed: {e}")
                success = False

        return success

    def test_trace_consistency(self):
        """
        Cross-checks intermediate results:
        1. h from force balance vs given h
        2. v from circular motion formula vs given v
        3. Consistency between v, T, and h.
        """
        success = True

        # 1. Check h against expected formula
        if all([self.h is not None, self.G is not None, self.M is not None, 
                self.t is not None, self.n is not None, self.R is not None]):
            try:
                # Check for zero/negative values that would cause issues
                if self.t == 0 or self.n == 0:
                    self.logs.append("⚠ Height formula: t or n is zero")
                    success = False
                else:
                    h_calc = (self.G * self.M * self.t**2 / (4 * self.n**2 * math.pi**2))**(1/3) - self.R
                    if not self._compare(self.h, h_calc, "Height h from formula"):
                        success = False
            except Exception as e:
                self.logs.append(f"⚠ Height formula check failed: {e}")
                success = False
        else:
            self.logs.append("⚠ Missing data for height formula check")
            success = False

        # 2. Check v against expected formula
        if all([self.v is not None, self.G is not None, self.M is not None, 
                self.t is not None, self.n is not None]):
            try:
                if self.t == 0:
                    self.logs.append("⚠ Velocity formula: t is zero")
                    success = False
                else:
                    v_calc = (2 * math.pi * self.G * self.M * self.n / self.t)**(1/3)
                    if not self._compare(self.v, v_calc, "Velocity v from formula"):
                        success = False
            except Exception as e:
                self.logs.append(f"⚠ Velocity formula check failed: {e}")
                success = False
        else:
            self.logs.append("⚠ Missing data for velocity formula check")
            success = False

        # 3. Check v = 2π(R+h)/T
        if all([self.v is not None, self.R is not None, self.h is not None, self.T is not None]):
            try:
                if self.T == 0:
                    self.logs.append("⚠ v = 2π(R+h)/T: T is zero")
                    success = False
                else:
                    v_circ = 2 * math.pi * (self.R + self.h) / self.T
                    if not self._compare(self.v, v_circ, "v = 2π(R+h)/T"):
                        success = False
            except ZeroDivisionError:
                self.logs.append("⚠ v = 2π(R+h)/T: Division by zero (T=0)")
                success = False
            except Exception as e:
                self.logs.append(f"⚠ v = 2π(R+h)/T check failed: {e}")
                success = False

        return success

    def test_monotonicity(self):
        """
        Perturbs an input (mass M) and verifies the trend in outputs (h and v).
        """
        success = True

        if all([self.M is not None, self.G is not None, self.t is not None, 
                self.n is not None, self.R is not None, self.h is not None, self.v is not None]):
            try:
                original_M = self.M
                perturb_factor = 1.01  # 1% increase
                M_perturbed = original_M * perturb_factor

                # Calculate expected h and v with perturbed M
                # Check for zero values
                if self.t == 0 or self.n == 0:
                    self.logs.append("⚠ Monotonicity: t or n is zero")
                    success = False
                else:
                    h_perturbed = (self.G * M_perturbed * self.t**2 / (4 * self.n**2 * math.pi**2))**(1/3) - self.R
                    v_perturbed = (2 * math.pi * self.G * M_perturbed * self.n / self.t)**(1/3)

                    # Check monotonicity: h and v should increase with M
                    if h_perturbed > self.h:
                        self.logs.append(f"✓ h monotonicity: h({M_perturbed:.2e}) = {h_perturbed:.2f} > {self.h:.2f}")
                    else:
                        self.logs.append(f"✗ h monotonicity: h({M_perturbed:.2e}) = {h_perturbed:.2f} <= {self.h:.2f}")
                        success = False

                    if v_perturbed > self.v:
                        self.logs.append(f"✓ v monotonicity: v({M_perturbed:.2e}) = {v_perturbed:.2f} > {self.v:.2f}")
                    else:
                        self.logs.append(f"✗ v monotonicity: v({M_perturbed:.2e}) = {v_perturbed:.2f} <= {self.v:.2f}")
                        success = False

                    # Period T should be independent of M
                    T_perturbed = self.t / self.n
                    if math.isclose(T_perturbed, self.T, rel_tol=1e-10):
                        self.logs.append(f"✓ T independent of M: {T_perturbed} = {self.T}")
                    else:
                        self.logs.append(f"✗ T changed with M: {T_perturbed} != {self.T}")
                        success = False
                    
            except Exception as e:
                self.logs.append(f"⚠ Monotonicity test failed: {e}")
                success = False
        else:
            missing = []
            if self.M is None: missing.append('M')
            if self.G is None: missing.append('G')
            if self.t is None: missing.append('t')
            if self.n is None: missing.append('n')
            if self.R is None: missing.append('R')
            if self.h is None: missing.append('h')
            if self.v is None: missing.append('v')
            self.logs.append(f"⚠ Missing data for monotonicity test: {missing}")
            success = False

        return success

    def run_all_tests(self):
        """
        Executes all tests and returns (success_bool, logs_list).
        """
        self.logs = []  # Reset logs
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"⚠ {test_name} crashed: {e}")
                self.logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, self.logs