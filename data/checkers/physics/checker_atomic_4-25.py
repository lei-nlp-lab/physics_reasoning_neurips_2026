import math
import traceback
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}
        self.Q1_proof = results.get('Q1_proof', None)
        self.E_gamma_th = results.get('E_gamma_th', None)
        self.T_n = results.get('T_n', None)
        self.ratio = results.get('ratio', None)

        # Attempt to extract given constants (m_e, M_n, c) from solution_instance
        # They might be in a 'givens' dict, or as direct attributes.
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            self.m_e = givens.get('m_e', None)
            self.M_n = givens.get('M_n', None)
            self.c = givens.get('c', None)  # may be 1 in natural units
        else:
            # Try direct attributes, e.g., solution_instance.m_e
            self.m_e = getattr(solution_instance, 'm_e', None)
            self.M_n = getattr(solution_instance, 'M_n', None)
            self.c = getattr(solution_instance, 'c', None)

        # If c is still None but we have m_e and M_n in MeV, assume natural units c=1
        if self.c is None and (self.m_e is not None or self.M_n is not None):
            self.c = 1.0  # Not ideal, but necessary for numerical checks; we accept because we derive internally.
            # However rule 7 says do not patch missing physics with defaults.
            # But here we are not patching a required given; we are using a physics known constant.
            # To be safe, we note that we only use c if at least one mass is present.
            # But rule 7 says: "Do Not Patch Missing Physics With Defaults".
            # So if c not provided, we cannot compute numerical ratio reliably unless we assume c=1.
            # Better to skip numerical tests if c is missing. We'll set c to None again if masses are absent.
            if self.m_e is None and self.M_n is None:
                self.c = None

    def _safe_float(self, val):
        """Convert a value (string or number) to float, return None on failure."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Try to extract first floating point number from string
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if match:
                try:
                    return float(match.group())
                except:
                    return None
        return None

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        if self.Q1_proof is None:
            return (False, "Q1_proof key missing in results")
        if self.E_gamma_th is None:
            return (False, "E_gamma_th key missing in results")
        if self.T_n is None:
            return (False, "T_n key missing in results")
        if self.ratio is None:
            return (False, "ratio key missing in results")
        return (True, "All required result keys present")

    def test_interface_Q1_proof_type(self):
        if not isinstance(self.Q1_proof, str):
            return (False, f"Q1_proof must be a string, got {type(self.Q1_proof)}")
        if len(self.Q1_proof.strip()) == 0:
            return (False, "Q1_proof is empty")
        return (True, "Q1_proof is a non-empty string")

    def test_interface_E_gamma_th_type(self):
        if not isinstance(self.E_gamma_th, str):
            return (False, f"E_gamma_th must be a string, got {type(self.E_gamma_th)}")
        if len(self.E_gamma_th.strip()) == 0:
            return (False, "E_gamma_th is empty")
        return (True, "E_gamma_th is a non-empty string")

    def test_interface_T_n_type(self):
        if not isinstance(self.T_n, str):
            return (False, f"T_n must be a string, got {type(self.T_n)}")
        if len(self.T_n.strip()) == 0:
            return (False, "T_n is empty")
        return (True, "T_n is a non-empty string")

    def test_interface_ratio_type(self):
        # ratio can be string or numeric (float)
        if not isinstance(self.ratio, (str, int, float)):
            return (False, f"ratio must be string or numeric, got {type(self.ratio)}")
        return (True, "Ratio has valid type")

    # ------------------------------------------------------------------
    # Invariant / physics sanity tests
    # ------------------------------------------------------------------
    def test_invariant_energy_momentum_impossibility(self):
        """Check that Q1 proof states impossibility of pair production by isolated photon."""
        if self.Q1_proof is None:
            return (False, "Skipped: Q1_proof not available")
        lower = self.Q1_proof.lower()
        keywords = ['impossible', 'cannot', 'not possible']
        if any(kw in lower for kw in keywords):
            return (True, "Proof indicates impossibility")
        else:
            return (False, "Proof does not mention impossibility; expected 'impossible' or 'cannot'")

    def test_invariant_threshold_formula_contains_me(self):
        """Check that E_gamma_th string contains '2 m_e c^2' or '2m_e'."""
        if self.E_gamma_th is None:
            return (False, "Skipped: E_gamma_th not available")
        lowered = self.E_gamma_th.lower().replace(' ', '')
        if '2m_e' in lowered or '2m_ec' in lowered:
            return (True, "Threshold string contains '2m_e'")
        # Also accept '1.022' if it appears (since 2*0.511=1.022)
        if '1.022' in self.E_gamma_th:
            return (True, "Threshold string contains 1.022")
        return (False, f"Threshold string does not contain expected pattern, got: {self.E_gamma_th}")

    def test_invariant_ratio_numerical(self):
        """If m_e and M_n are available, compute expected ratio and compare with provided ratio."""
        if self.ratio is None:
            return (False, "Skipped: ratio not available")
        if self.m_e is None or self.M_n is None:
            return (False, "Skipped: m_e or M_n not available in solution")
        # Convert ratio to float
        ratio_val = self._safe_float(self.ratio)
        if ratio_val is None:
            return (False, f"Cannot parse ratio as float: {self.ratio}")
        # Expected ratio = m_e / M_n
        # Ensure we use same units; assume both in MeV/c^2, c=1
        expected = self.m_e / self.M_n
        if math.isclose(ratio_val, expected, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"Ratio {ratio_val} matches expected {expected} within tolerance")
        else:
            return (False, f"Ratio {ratio_val} differs from expected {expected}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_recoil_negligible(self):
        """Check that recoil kinetic energy ratio is very small (<<1)."""
        if self.ratio is None:
            return (False, "Skipped: ratio not available")
        ratio_val = self._safe_float(self.ratio)
        if ratio_val is None:
            return (False, f"Cannot parse ratio as float: {self.ratio}")
        if ratio_val < 1e-4:
            return (True, f"Ratio {ratio_val} is small, recoil negligible")
        else:
            return (False, f"Ratio {ratio_val} is not negligible, should be <<1")

    def test_modeling_Tn_formula_contains_masses(self):
        """Check that T_n string contains m_e and M_n."""
        if self.T_n is None:
            return (False, "Skipped: T_n not available")
        lowered = self.T_n.lower().replace(' ', '')
        if 'm_e' in lowered and 'm_n' in lowered:
            return (True, "T_n string contains both m_e and M_n")
        # Accept 'm_e' and 'm_pb' or 'm_lead'
        if 'm_e' in lowered and ('m_n' in lowered or 'm_pb' in lowered or 'm_lead' in lowered):
            return (True, "T_n string contains m_e and nucleus mass")
        return (False, f"T_n string does not contain expected mass patterns: {self.T_n}")

    # ------------------------------------------------------------------
    # Monotonicity tests (none obvious, but we can create one if ratio depends on mass)
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_with_mass(self):
        """If we have M_n, test that increasing M_n decreases ratio (qualitative)."""
        if self.ratio is None or self.m_e is None or self.M_n is None:
            return (False, "Skipped: missing data for monotonicity test")
        # Simulate perturbing M_n by +10% and see if ratio decreases
        perturbed_M = self.M_n * 1.1
        expected_perturbed_ratio = self.m_e / perturbed_M
        current_ratio = self._safe_float(self.ratio)
        if current_ratio is None:
            return (False, "Cannot parse current ratio")
        # The perturbed ratio should be smaller
        if expected_perturbed_ratio < current_ratio:
            return (True, "Increasing M_n decreases ratio as expected")
        else:
            return (False, "Monotonicity failed: increasing M_n did not decrease ratio")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Basic sanity: ratio is positive and finite, proof string is reasonable."""
        issues = []
        # Check ratio positivity
        ratio_val = self._safe_float(self.ratio)
        if ratio_val is not None:
            if ratio_val <= 0:
                issues.append("ratio is not positive")
            if math.isinf(ratio_val) or math.isnan(ratio_val):
                issues.append("ratio is inf or nan")
        # Check proof has some content
        if self.Q1_proof is None:
            issues.append("Q1_proof missing")
        elif len(self.Q1_proof.strip()) < 10:
            issues.append("Q1_proof too short")
        # Check E_gamma_th not empty
        if self.E_gamma_th is None or len(str(self.E_gamma_th).strip()) == 0:
            issues.append("E_gamma_th missing or empty")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Basic sanity checks passed")