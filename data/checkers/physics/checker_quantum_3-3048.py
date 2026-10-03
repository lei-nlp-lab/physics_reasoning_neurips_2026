import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.pseudocode = getattr(solution_instance, 'pseudocode', '')
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', '')
        self.givens = getattr(solution_instance, 'givens', {})

        # Extract required parameters with safe defaults (None)
        self.a = getattr(solution_instance, 'a', None)
        self.B = getattr(solution_instance, 'B', None)
        self.m_e = getattr(solution_instance, 'm_e', None)
        self.m_mu = getattr(solution_instance, 'm_mu', None)
        self.e = getattr(solution_instance, 'e', None)
        self.c = getattr(solution_instance, 'c', None)

        # Fallback to givens dict
        if self.a is None:
            self.a = self.givens.get('a', None)
        if self.B is None:
            self.B = self.givens.get('B', None)
        if self.m_e is None:
            self.m_e = self.givens.get('m_e', None)
        if self.m_mu is None:
            self.m_mu = self.givens.get('m_mu', None)
        if self.e is None:
            self.e = self.givens.get('e', None)
        if self.c is None:
            self.c = self.givens.get('c', None)

        # Extract specific result keys
        self.dominant_term = self.results.get('dominant_term', None)
        self.E_11 = self.results.get('E_11', None)
        self.detection_method = self.results.get('detection_method', None)

        # Store explanation and pseudocode as strings for easy searching
        self.explanation = self.explanation_steps if isinstance(self.explanation_steps, str) else ''
        self.pseudo_str = self.pseudocode if isinstance(self.pseudocode, str) else ''

    # --- Helper: compute expected E_11 from parameters ---
    def _expected_E11(self, a_val, B_val, m_e_val, m_mu_val, e_val, c_val):
        """Compute theoretical E_11 eigenvalue using the formula from the pseudocode."""
        return (a_val / 4.0 +
                (e_val * B_val) / (2.0 * m_e_val * c_val) -
                (e_val * B_val) / (2.0 * m_mu_val * c_val))

    # --- Interface tests ---
    def test_interface_required_keys(self):
        if 'dominant_term' not in self.results:
            return (False, "Missing 'dominant_term' in results")
        if 'E_11' not in self.results:
            return (False, "Missing 'E_11' in results")
        if 'detection_method' not in self.results:
            return (False, "Missing 'detection_method' in results")
        return (True, "All required keys present")

    def test_interface_dominant_term_type(self):
        if self.dominant_term is None:
            return (False, "dominant_term is None")
        if not isinstance(self.dominant_term, str):
            return (False, f"dominant_term is not a string, got {type(self.dominant_term)}")
        return (True, "dominant_term is a string")

    def test_interface_E11_type(self):
        if self.E_11 is None:
            return (False, "E_11 is None")
        if not isinstance(self.E_11, (int, float)):
            return (False, f"E_11 is not a number, got {type(self.E_11)}")
        if isinstance(self.E_11, complex):
            return (False, "E_11 should be real, not complex")
        return (True, "E_11 is a real number")

    def test_interface_detection_method_type(self):
        if self.detection_method is None:
            return (False, "detection_method is None")
        if not isinstance(self.detection_method, str):
            return (False, f"detection_method is not a string, got {type(self.detection_method)}")
        return (True, "detection_method is a string")

    # --- Invariant / theorem-based checks ---
    def test_invariant_hyperfine_energy_formula(self):
        # Check that E_11 matches the formula a/4 + eB/(2 m_e c) - eB/(2 m_mu c)
        if self.E_11 is None:
            return (False, "E_11 not available")
        if any(p is None for p in [self.a, self.B, self.m_e, self.m_mu, self.e, self.c]):
            return (False, "Skip: missing parameters for numeric check")
        try:
            expected = self._expected_E11(self.a, self.B, self.m_e, self.m_mu, self.e, self.c)
            if math.isclose(self.E_11, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E_11 matches formula: {self.E_11} ≈ {expected}")
            else:
                return (False, f"E_11 = {self.E_11} does not match expected {expected}")
        except Exception as e:
            return (False, f"Numeric check crashed: {e}")

    def test_invariant_zeeman_sign_in_explanation(self):
        if not self.explanation:
            return (False, "Skip: no explanation steps provided")
        if "negative" in self.explanation.lower() or "opposite sign" in self.explanation.lower():
            return (True, "Explanation correctly identifies negative muon Zeeman term")
        else:
            return (False, "Explanation does not mention negative sign of muon Zeeman term")

    def test_invariant_hyperfine_eigenvalue_value(self):
        # Check that the trace mentions the hyperfine eigenvalue 1/4 for |1,+1>
        found = False
        if "1/4" in self.explanation or "0.25" in self.explanation:
            found = True
        if "1/4" in self.pseudo_str or "0.25" in self.pseudo_str:
            found = True
        if not found:
            return (False, "Neither explanation nor pseudocode mentions hyperfine eigenvalue 1/4")
        return (True, "Hyperfine eigenvalue 1/4 is mentioned in trace")

    # --- Modeling / trace checks ---
    def test_modeling_state_used(self):
        if "|1,+1>" in self.explanation or "|1, +1>" in self.explanation or "(1, +1)" in self.explanation:
            return (True, "State |1,+1> is mentioned in explanation")
        if "|1,+1>" in self.pseudo_str or "|1, +1>" in self.pseudo_str:
            return (True, "State |1,+1> is mentioned in pseudocode")
        return (False, "State |1,+1> not found in traces")

    def test_modeling_hyperfine_eigenvalue_in_explanation(self):
        if "a/4" in self.explanation:
            return (True, "Explanation gives hyperfine eigenvalue a/4")
        return (False, "Explanation does not contain 'a/4'")

    def test_trace_pseudocode_uses_hyperfine_relation(self):
        if "F^2 - 3/2" in self.pseudo_str or "F^2 - 1.5" in self.pseudo_str:
            return (True, "Pseudocode uses hyperfine relation")
        return (False, "Pseudocode missing hyperfine relation")

    def test_trace_pseudocode_uses_zeeman_terms(self):
        if "Omega_e" in self.pseudo_str and "Omega_mu" in self.pseudo_str:
            return (True, "Pseudocode defines Zeeman frequencies")
        return (False, "Pseudocode missing Zeeman terms")

    # --- Monotonicity / trend checks (explicit perturbation of an input) ---
    def test_monotonicity_E11_with_B(self):
        """Perturb B by +10% and check that the expected E_11 increases (electron Zeeman dominates)."""
        if any(p is None for p in [self.a, self.B, self.m_e, self.m_mu, self.e, self.c]):
            return (False, "Skip: missing parameters for monotonicity check")
        try:
            B0 = self.B
            B1 = 1.1 * B0   # perturb B upward by 10%
            E0 = self._expected_E11(self.a, B0, self.m_e, self.m_mu, self.e, self.c)
            E1 = self._expected_E11(self.a, B1, self.m_e, self.m_mu, self.e, self.c)
            if E1 > E0:
                return (True, f"E_11 increases with B: baseline E0={E0:.6f}, perturbed E1={E1:.6f}")
            else:
                return (False, f"E_11 does not increase with B: baseline E0={E0:.6f}, perturbed E1={E1:.6f}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    def test_monotonicity_E11_with_a(self):
        """Perturb a by +10% and check that E_11 increases (hyperfine term is positive)."""
        if any(p is None for p in [self.a, self.B, self.m_e, self.m_mu, self.e, self.c]):
            return (False, "Skip: missing parameters for monotonicity check")
        try:
            a0 = self.a
            a1 = 1.1 * a0   # perturb a upward by 10%
            E0 = self._expected_E11(a0, self.B, self.m_e, self.m_mu, self.e, self.c)
            E1 = self._expected_E11(a1, self.B, self.m_e, self.m_mu, self.e, self.c)
            if E1 > E0:
                return (True, f"E_11 increases with a: baseline E0={E0:.6f}, perturbed E1={E1:.6f}")
            else:
                return (False, f"E_11 does not increase with a: baseline E0={E0:.6f}, perturbed E1={E1:.6f}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    # --- Weak answer sanity check (at most one) ---
    def test_answer_sanity_basic(self):
        if self.dominant_term is None:
            return (False, "dominant_term is None")
        if not isinstance(self.dominant_term, str) or len(self.dominant_term) == 0:
            return (False, "dominant_term is empty or not a string")
        if self.E_11 is None:
            return (False, "E_11 is None")
        if not isinstance(self.E_11, (int, float)):
            return (False, "E_11 is not a number")
        if not math.isfinite(self.E_11):
            return (False, f"E_11 is not finite: {self.E_11}")
        if self.detection_method is None:
            return (False, "detection_method is None")
        if not isinstance(self.detection_method, str) or len(self.detection_method) == 0:
            return (False, "detection_method is empty or not a string")
        return (True, "All three results present and pass basic sanity checks")

    # --- Runner ---
    def run_all_tests(self):
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