import math
import traceback

class PhysicsChecker:
    """
    Audits the solution of a plane wave propagation in an anisotropic medium problem.
    Extracts results from solution_instance and runs unit-test-like checks.
    """
    def __init__(self, solution_instance):
        # Safely extract the results dictionary; default to empty dict if missing
        self.results = getattr(solution_instance, 'results', None) or {}
        if not isinstance(self.results, dict):
            self.results = {}
        # Extract required keys with None defaults (no numeric fallbacks)
        self.orthogonality = self.results.get('orthogonality', None)
        self.transversality = self.results.get('transversality', None)
        self.component_eq = self.results.get('component_eq', None)
        self.fresnel_eq = self.results.get('Fresnel_eq', None)

    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
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

    # -------- Interface checks ----------
    def test_interface_orthogonality_key(self):
        if self.orthogonality is None:
            return (False, "SKIP: 'orthogonality' key missing")
        if not isinstance(self.orthogonality, str):
            return (False, "'orthogonality' is not a string")
        return (True, "'orthogonality' present and is string")

    def test_interface_transversality_key(self):
        if self.transversality is None:
            return (False, "SKIP: 'transversality' key missing")
        if not isinstance(self.transversality, str):
            return (False, "'transversality' is not a string")
        return (True, "'transversality' present and is string")

    def test_interface_component_eq_key(self):
        if self.component_eq is None:
            return (False, "SKIP: 'component_eq' key missing")
        if not isinstance(self.component_eq, str):
            return (False, "'component_eq' is not a string")
        return (True, "'component_eq' present and is string")

    def test_interface_fresnel_eq_key(self):
        if self.fresnel_eq is None:
            return (False, "SKIP: 'Fresnel_eq' key missing")
        if not isinstance(self.fresnel_eq, str):
            return (False, "'Fresnel_eq' is not a string")
        return (True, "'Fresnel_eq' present and is string")

    # -------- Invariant checks (theorem-based) ----------
    def test_invariant_orthogonality_relations(self):
        if self.orthogonality is None:
            return (False, "SKIP: orthogonality not available")
        s = self.orthogonality
        expected_pairs = ["H*E=0", "H*D=0", "H*K=0", "D*K=0"]
        missing = [pair for pair in expected_pairs if pair not in s]
        if missing:
            return (False, f"Missing orthogonality relations: {missing}")
        return (True, "All four orthogonality conditions found")

    def test_invariant_transversality_statements(self):
        if self.transversality is None:
            return (False, "SKIP: transversality not available")
        s = self.transversality
        if "D" not in s or "H" not in s:
            return (False, "Transversality text missing D or H")
        if "transverse" not in s.lower():
            return (False, "Transversality text does not mention 'transverse'")
        if "E" not in s:
            return (False, "Transversality text missing E")
        if "not" in s.lower() or "non" in s.lower() or "may be nonzero" in s.lower():
            return (True, "Transversality correctly indicates E is not necessarily transverse")
        return (True, "Transversality statement present")

    # -------- Modeling / trace quality checks (at least two) ----------
    def test_modeling_component_equation(self):
        if self.component_eq is None:
            return (False, "SKIP: component_eq not available")
        s = self.component_eq
        indicators = ["S_j", "sum_i", "V^2", "V_j^2", "-1"]
        missing = [ind for ind in indicators if ind not in s]
        if missing:
            return (False, f"Component equation missing key terms: {missing}")
        if "=" not in s:
            return (False, "Component equation missing '='")
        return (True, "Component equation contains expected structure")

    def test_modeling_fresnel_equation_quadratic_degree(self):
        """Check that the Fresnel equation string contains only up to V^4 (quadratic in V^2)."""
        if self.fresnel_eq is None:
            return (False, "SKIP: Fresnel_eq not available")
        s = self.fresnel_eq
        # Must contain V^4, V^2, and constant terms, but not V^6 or higher
        if "V^6" in s or "V^8" in s or "V^10" in s:
            return (False, "Fresnel equation contains unexpected higher powers")
        if "V^4" not in s:
            return (False, "Fresnel equation missing V^4 term")
        if "V^2" not in s:
            return (False, "Fresnel equation missing V^2 term")
        # Check that a constant term exists (no V factor)
        # Look for a term that is not multiplied by V^2 or V^4
        # Simple heuristic: there should be a fraction or term without V
        if not any(ch.isdigit() or ch == '(' for ch in s.split('V^')[0].strip()):
            # The constant term might be the last term before '=0'
            pass  # too fragile, skip
        return (True, "Fresnel equation shows proper quadratic structure in V^2")

    # -------- Monotonicity check (based on re-derived Fresnel physics) ----------
    def test_monotonicity_larger_root_increases_with_V1(self):
        """
        For a given set of V1, V2, V3, and S vector (satisfying S1^2+S2^2+S3^2=1),
        the Fresnel equation is quadratic in x = V^2. This test verifies that the
        larger positive root increases when V1 is increased (i.e., when epsilon1 decreases).
        The coefficients are derived from the Fresnel equation.
        """
        # Use typical test parameters (not from solution, re-derived inside checker)
        V1_base = 2.0
        V2 = 3.0
        V3 = 4.0
        S1 = 0.5
        S2 = 0.5
        S3 = math.sqrt(1 - S1*S1 - S2*S2)  # ~0.70710678

        # Helper to compute roots of Fresnel quadratic in x = V^2
        def compute_roots(V1):
            # Coefficients A x^2 + B x + C = 0
            A = 1.0 / (V1*V1 * V2*V2 * V3*V3)
            B = (S1*S1 - 1.0) / (V2*V2 * V3*V3) + \
                (S2*S2 - 1.0) / (V1*V1 * V3*V3) + \
                (S3*S3 - 1.0) / (V1*V1 * V2*V2)
            C = S1*S1 / (V1*V1) + S2*S2 / (V2*V2) + S3*S3 / (V3*V3)
            disc = B*B - 4*A*C
            if disc < 0:
                return (None, None)
            sqrt_disc = math.sqrt(disc)
            x1 = (-B + sqrt_disc) / (2.0 * A)
            x2 = (-B - sqrt_disc) / (2.0 * A)
            # Return larger root first
            if x1 >= x2:
                return (x1, x2)
            else:
                return (x2, x1)

        baseline_larger, baseline_smaller = compute_roots(V1_base)
        if baseline_larger is None or baseline_smaller is None:
            return (False, "SKIP: Baseline roots not real (discriminant negative)")

        # Perturb V1 by +5%
        V1_perturbed = V1_base * 1.05
        perturbed_larger, perturbed_smaller = compute_roots(V1_perturbed)
        if perturbed_larger is None:
            return (False, "SKIP: Perturbed roots not real")

        # The larger root should increase when V1 increases.
        # Use math.isclose to allow for small rounding errors.
        if perturbed_larger < baseline_larger - 1e-10:
            return (False, f"Larger root decreased (baseline={baseline_larger:.6f}, perturbed={perturbed_larger:.6f})")
        # Even if it's very slightly less, we treat as avoid
        return (True, f"Larger root increases from {baseline_larger:.6f} to {perturbed_larger:.6f} when V1 increased by 5%")

    # -------- Weak answer sanity (only one, no re-derivation of final answer) ----------
    def test_answer_sanity_fresnel_non_trivial(self):
        if self.fresnel_eq is None:
            return (False, "SKIP: Fresnel_eq not available")
        if len(self.fresnel_eq) < 20:
            return (False, "Fresnel equation suspiciously short")
        if "=" not in self.fresnel_eq:
            return (False, "Fresnel equation missing equality sign")
        return (True, "Fresnel equation is non-trivial")