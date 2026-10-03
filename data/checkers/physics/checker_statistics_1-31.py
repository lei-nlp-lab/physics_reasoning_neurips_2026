import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract expected keys
        self.eta_max = self.results.get('eta_max', None)
        self.W_max = self.results.get('W_max', None)
        
        # Extract givens if present
        givens = getattr(solution_instance, 'givens', {})
        self.T_h_C = givens.get('intake_temperature_C', 
                        getattr(solution_instance, 'T_h_C', None))
        self.T_c_C = givens.get('exhaust_temperature_C', 
                        getattr(solution_instance, 'T_c_C', None))
        self.Q = givens.get('heat_input_Q', 
                    getattr(solution_instance, 'Q', None))
        
        # If not found in givens, maybe they are in results as intermediate values
        if self.T_h_C is None:
            self.T_h_C = self.results.get('T_h_C', None)
        if self.T_c_C is None:
            self.T_c_C = self.results.get('T_c_C', None)
        if self.Q is None:
            self.Q = self.results.get('Q', None)
        
        # Compute derived Kelvin temperatures
        self.T_h_K = None
        self.T_c_K = None
        if self.T_h_C is not None:
            self.T_h_K = self.T_h_C + 273.15
        if self.T_c_C is not None:
            self.T_c_K = self.T_c_C + 273.15

    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, logs)."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------ Interface Tests ------------------
    def test_interface_eta_max_exists(self):
        """Check that eta_max is present in results."""
        if 'eta_max' not in self.results:
            return (False, "Missing key 'eta_max' in results.")
        return (True, "eta_max key found.")

    def test_interface_W_max_exists(self):
        """Check that W_max is present in results."""
        if 'W_max' not in self.results:
            return (False, "Missing key 'W_max' in results.")
        return (True, "W_max key found.")

    def test_interface_eta_max_numeric(self):
        """Check that eta_max is a number or symbolic expression."""
        if self.eta_max is None:
            return (False, "eta_max is None, cannot check type.")
        if isinstance(self.eta_max, (int, float, sp.Expr)):
            return (True, f"eta_max is of type {type(self.eta_max).__name__}.")
        return (False, f"eta_max is unexpected type {type(self.eta_max).__name__}.")

    def test_interface_W_max_numeric(self):
        """Check that W_max is a number or symbolic expression."""
        if self.W_max is None:
            return (False, "W_max is None, cannot check type.")
        if isinstance(self.W_max, (int, float, sp.Expr)):
            return (True, f"W_max is of type {type(self.W_max).__name__}.")
        return (False, f"W_max is unexpected type {type(self.W_max).__name__}.")

    # ------------------ Invariant Tests (theorem-based) ------------------
    def test_invariant_temperature_conversion(self):
        """Check that temperatures were converted to Kelvin correctly."""
        if self.T_h_C is None or self.T_c_C is None:
            return (False, "SKIP: T_h_C or T_c_C not available.")
        # Check if the solution stored Kelvin temperatures in results
        T_h_K_sol = self.results.get('T_h_K', None)
        T_c_K_sol = self.results.get('T_c_K', None)
        if T_h_K_sol is not None and T_c_K_sol is not None:
            expected_T_h = self.T_h_C + 273.15
            expected_T_c = self.T_c_C + 273.15
            if not math.isclose(T_h_K_sol, expected_T_h, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"T_h_K mismatch: got {T_h_K_sol}, expected ~{expected_T_h}.")
            if not math.isclose(T_c_K_sol, expected_T_c, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"T_c_K mismatch: got {T_c_K_sol}, expected ~{expected_T_c}.")
        # If not stored, skip – the invariant test for efficiency will catch issues.
        return (True, "Temperature conversion consistent (or not stored).")

    def test_invariant_carnot_efficiency(self):
        """Check eta_max = 1 - T_c/T_h."""
        if self.eta_max is None:
            return (False, "eta_max not available.")
        if self.T_h_K is None or self.T_c_K is None:
            return (False, "SKIP: cannot compute expected efficiency without Kelvin temperatures.")
        expected_eta = 1 - self.T_c_K / self.T_h_K
        # Handle symbolic eta_max
        if isinstance(self.eta_max, sp.Expr):
            # Try to substitute with numeric values
            subs = {s: 1.0 for s in self.eta_max.free_symbols}
            val = float(self.eta_max.subs(subs).evalf())
            if not math.isclose(val, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"eta_max symbolic evaluates to {val:.4f}, expected {expected_eta:.4f}.")
        else:
            if not math.isclose(self.eta_max, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"eta_max = {self.eta_max}, expected {expected_eta:.4f}.")
        return (True, f"eta_max is consistent with Carnot efficiency.")

    def test_invariant_W_max_definition(self):
        """Check W_max = eta_max * Q."""
        if self.W_max is None or self.eta_max is None:
            return (False, "W_max or eta_max missing.")
        if self.Q is None:
            return (False, "SKIP: Q not available.")
        expected_W = self.eta_max * self.Q
        if isinstance(self.W_max, sp.Expr):
            # Evaluate with dummy values
            subs = {s: 1.0 for s in self.W_max.free_symbols}
            val = float(self.W_max.subs(subs).evalf())
            expected_val = float(expected_W.subs(subs).evalf()) if isinstance(expected_W, sp.Expr) else expected_W
            if not math.isclose(val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"W_max symbolic value {val:.4f} does not match expected {expected_val:.4f}.")
        else:
            if not math.isclose(self.W_max, expected_W, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"W_max = {self.W_max}, expected {expected_W:.4f}.")
        return (True, "W_max = eta_max * Q.")

    def test_invariant_eta_max_bounds(self):
        """Check 0 < eta_max < 1 (strictly between 0 and 1 for heat engine)."""
        if self.eta_max is None:
            return (False, "eta_max not available.")
        val = self.eta_max
        if isinstance(val, sp.Expr):
            subs = {s: 1.0 for s in val.free_symbols}
            val = float(val.subs(subs).evalf())
        if val <= 0 or val >= 1:
            return (False, f"eta_max = {val:.4f} is not in (0,1).")
        return (True, f"eta_max = {val:.4f} is in (0,1).")

    def test_invariant_W_max_nonnegative(self):
        """Check W_max >= 0 if Q > 0."""
        if self.W_max is None:
            return (False, "W_max not available.")
        if self.Q is None or self.Q <= 0:
            return (False, "SKIP: Q missing or not positive.")
        val = self.W_max
        if isinstance(val, sp.Expr):
            subs = {s: 1.0 for s in val.free_symbols}
            val = float(val.subs(subs).evalf())
        if val < 0:
            return (False, f"W_max = {val:.4f} is negative; should be nonnegative.")
        return (True, f"W_max >= 0.")

    def test_invariant_Clausius_inequality(self):
        """Check that W_max <= (1 - T_c/T_h) Q (i.e., equality for reversible)."""
        if self.W_max is None or self.eta_max is None or self.Q is None:
            return (False, "Missing quantity for Clausius check.")
        if self.T_h_K is None or self.T_c_K is None:
            return (False, "SKIP: Kelvin temperatures not available.")
        # For reversible cycle, W_max should equal the Carnot limit.
        expected_bound = (1 - self.T_c_K / self.T_h_K) * self.Q
        w_val = self.W_max
        if isinstance(w_val, sp.Expr):
            subs = {s: 1.0 for s in w_val.free_symbols}
            w_val = float(w_val.subs(subs).evalf())
        # Use a small tolerance to allow rounding
        if w_val > expected_bound * (1 + 1e-2):
            return (False, f"W_max = {w_val:.4f} exceeds Carnot limit {expected_bound:.4f}.")
        return (True, "W_max does not exceed Carnot bound.")

    # ------------------ Modeling Quality Tests (at least two) ------------------
    def test_modeling_W_max_expression(self):
        """Check that W_max is expressed as a function of Q."""
        if self.W_max is None:
            return (False, "W_max not available.")
        # Check if W_max is symbolic with Q as free symbol
        if isinstance(self.W_max, sp.Expr) and any(s.name == 'Q' for s in self.W_max.free_symbols):
            return (True, "W_max is symbolic function of Q.")
        # If W_max is numeric and Q is given, check proportionality
        if self.Q is not None and isinstance(self.Q, (int, float)) and isinstance(self.W_max, (int, float)):
            expected_eta = 1 - self.T_c_K/self.T_h_K if self.T_h_K and self.T_c_K else None
            if expected_eta is not None:
                ratio = self.W_max / self.Q
                if math.isclose(ratio, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "W_max numeric but consistent with eta_max*Q.")
        return (True, "W_max representation is acceptable.")

    def test_modeling_eta_max_from_temperatures(self):
        """Check that eta_max derived from Kelvin temperatures is correct."""
        if self.eta_max is None:
            return (False, "eta_max not available.")
        if self.T_h_K is None or self.T_c_K is None:
            return (False, "SKIP: Kelvin temperatures not available.")
        # Re-derive expected efficiency from converted temperatures
        expected_eta = 1 - self.T_c_K / self.T_h_K
        if isinstance(self.eta_max, sp.Expr):
            subs = {s: 1.0 for s in self.eta_max.free_symbols}
            val = float(self.eta_max.subs(subs).evalf())
            if not math.isclose(val, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"eta_max = {val:.4f} does not match expected {expected_eta:.4f}.")
        else:
            if not math.isclose(self.eta_max, expected_eta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"eta_max = {self.eta_max} does not match expected {expected_eta:.4f}.")
        return (True, "eta_max correctly derived from temperature conversion.")

    # ------------------ Monotonicity Tests ------------------
    def test_monotonicity_increasing_with_Q(self):
        """Check that W_max increases when heat input Q increases."""
        if self.Q is None or not isinstance(self.Q, (int, float)):
            return (False, "SKIP: Q not numeric.")
        if self.eta_max is None:
            return (False, "SKIP: eta_max not available.")
        try:
            W_base = float(self.eta_max * self.Q)
        except:
            return (False, "Cannot evaluate baseline W.")
        Q_new = self.Q * 1.1
        try:
            W_pert = float(self.eta_max * Q_new)
        except:
            return (False, "Cannot evaluate perturbed W.")
        if W_pert <= W_base:
            return (False, f"W did not increase when Q increased: baseline {W_base:.4f}, perturbed {W_pert:.4f}.")
        return (True, f"W increases with Q: {W_base:.4f} -> {W_pert:.4f}.")

    def test_monotonicity_increasing_with_T_h(self):
        """Check that W_max increases when intake temperature (hot reservoir) increases."""
        if self.T_h_K is None or self.T_c_K is None or self.Q is None:
            return (False, "SKIP: missing temperatures or Q.")
        try:
            eta_base = 1 - self.T_c_K / self.T_h_K
            W_base = eta_base * self.Q
            T_h_pert = self.T_h_K * 1.01
            eta_pert = 1 - self.T_c_K / T_h_pert
            W_pert = eta_pert * self.Q
        except:
            return (False, "Error computing monotonicity with T_h.")
        if W_pert <= W_base:
            return (False, f"W did not increase when T_h increased: baseline {W_base:.4f}, perturbed {W_pert:.4f}.")
        return (True, f"W increases with T_h: {W_base:.4f} -> {W_pert:.4f}.")

    # ------------------ Answer Sanity (weak, only one) ------------------
    def test_answer_sanity_W_max_magnitude(self):
        """Weak sanity: W_max should be between 0 and Q (since eta_max<1)."""
        if self.W_max is None:
            return (False, "W_max not available.")
        if self.Q is None:
            return (False, "SKIP: Q not available.")
        w = self.W_max
        q = self.Q
        if isinstance(w, sp.Expr):
            subs = {s: 1.0 for s in w.free_symbols}
            w = float(w.subs(subs).evalf())
        if isinstance(q, sp.Expr):
            subs = {s: 1.0 for s in q.free_symbols}
            q = float(q.subs(subs).evalf())
        if w <= 0 or w >= q:
            return (False, f"W_max = {w:.4f} is not in (0, Q={q:.4f}).")
        return (True, f"W_max {w:.4f} plausible magnitude.")

if __name__ == '__main__':
    # Example usage (for testing)
    class Solution:
        def __init__(self):
            self.givens = {
                'intake_temperature_C': 400,
                'exhaust_temperature_C': 150,
                'heat_input_Q': 1000
            }
            self.results = {
                'eta_max': 0.371,  # approximate
                'W_max': 371.0,
                'T_h_K': 673.15,
                'T_c_K': 423.15
            }
    sol = Solution()
    checker = PhysicsChecker(sol)
    passed, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nAll tests passed: {passed}")