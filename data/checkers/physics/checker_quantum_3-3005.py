import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem: find normalized eigenfunction of ŝ_y with eigenvalue -1/2.
    """
    def __init__(self, solution_instance):
        # Safe extraction of results from solution_instance
        # Expected keys: 'a', 'b', 'psi'
        self.a = None
        self.b = None
        self.psi = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.a = solution_instance.results.get('a', None)
            self.b = solution_instance.results.get('b', None)
            self.psi = solution_instance.results.get('psi', None)

        # Optional: try extracting explanation steps for modeling checks
        self.explanation_steps = None
        if hasattr(solution_instance, 'explanation_steps') and isinstance(solution_instance.explanation_steps, list):
            self.explanation_steps = solution_instance.explanation_steps

    # ------------------------------------------------------------
    # 1. Interface tests: required keys with valid types
    # ------------------------------------------------------------
    def test_interface_a_exists(self):
        if self.a is None:
            return (False, "Missing 'a' in results")
        return (True, "'a' is present")

    def test_interface_b_exists(self):
        if self.b is None:
            return (False, "Missing 'b' in results")
        return (True, "'b' is present")

    def test_interface_psi_exists(self):
        if self.psi is None:
            return (False, "Missing 'psi' in results")
        return (True, "'psi' is present")

    def test_interface_a_type(self):
        if self.a is None:
            return (True, "SKIP: 'a' not available")
        try:
            complex(self.a)
        except (TypeError, ValueError):
            return (False, f"'a' is not a complex number: {self.a}")
        return (True, "'a' is complex")

    def test_interface_b_type(self):
        if self.b is None:
            return (True, "SKIP: 'b' not available")
        try:
            complex(self.b)
        except (TypeError, ValueError):
            return (False, f"'b' is not a complex number: {self.b}")
        return (True, "'b' is complex")

    def test_interface_psi_type(self):
        if self.psi is None:
            return (True, "SKIP: 'psi' not available")
        if not isinstance(self.psi, (list, tuple)) or len(self.psi) != 2:
            return (False, f"'psi' must be a list/tuple of length 2, got {type(self.psi)}")
        try:
            complex(self.psi[0]), complex(self.psi[1])
        except (TypeError, ValueError):
            return (False, f"'psi' components are not complex numbers: {self.psi}")
        return (True, "'psi' is a 2-component complex vector")

    # ------------------------------------------------------------
    # 2. Invariant tests: physical laws using fuzzy matching
    # ------------------------------------------------------------
    def test_invariant_sy_eigenvalue(self):
        """
        Check that ŝ_y |ψ> = -(1/2) |ψ>
        ℏ = 1 implicit.
        """
        if self.psi is None or self.a is None or self.b is None:
            return (True, "SKIP: insufficient data for eigenvalue check")
        try:
            psi = [complex(self.psi[0]), complex(self.psi[1])]
            a, b = psi
        except (TypeError, IndexError):
            return (False, "Cannot interpret psi components as complex numbers")

        # sy = (1/2) * [[0, -i], [i, 0]]
        sy00 = 0.0
        sy01 = -0.5j
        sy10 = 0.5j
        sy11 = 0.0

        # Compute sy * psi
        res0 = sy00 * a + sy01 * b
        res1 = sy10 * a + sy11 * b

        # Expected: -0.5 * psi
        expected0 = -0.5 * a
        expected1 = -0.5 * b

        if not (math.isclose(res0.real, expected0.real, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(res0.imag, expected0.imag, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(res1.real, expected1.real, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(res1.imag, expected1.imag, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, f"Eigenvalue equation not satisfied: sy*psi = ({res0},{res1}), expected ({expected0},{expected1})")
        return (True, "Eigenvalue equation satisfied")

    def test_invariant_component_relation(self):
        """
        From the eigenvector equation we derive a = i * b.
        """
        if self.a is None or self.b is None:
            return (True, "SKIP: a or b not available")
        try:
            a = complex(self.a)
            b = complex(self.b)
        except (TypeError, ValueError):
            return (False, "a or b not complex")
        expected_a = 1j * b
        if not (math.isclose(a.real, expected_a.real, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(a.imag, expected_a.imag, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, f"Relation a = i*b not satisfied: a={a}, b={b}, i*b={expected_a}")
        return (True, "Component relation a = i*b holds")

    def test_invariant_normalization(self):
        """
        |a|^2 + |b|^2 = 1.
        """
        if self.a is None or self.b is None:
            return (True, "SKIP: a or b not available")
        try:
            a = complex(self.a)
            b = complex(self.b)
        except (TypeError, ValueError):
            return (False, "a or b not complex")
        norm = a.real**2 + a.imag**2 + b.real**2 + b.imag**2
        if not math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Normalization condition fails: |a|^2+|b|^2 = {norm}, expected 1.0")
        return (True, "Normalization condition satisfied")

    # ------------------------------------------------------------
    # 3. Modeling / trace tests: consistency with explanation steps
    # ------------------------------------------------------------
    def test_modeling_phase_convention(self):
        """
        In the explanation, b is chosen real positive. Check that b is real and positive.
        """
        if self.b is None:
            return (True, "SKIP: b not available")
        try:
            b = complex(self.b)
        except (TypeError, ValueError):
            return (False, "b is not complex")
        if not math.isclose(b.imag, 0.0, abs_tol=1e-2):
            return (False, f"Phase convention violated: b should be real, but imag part = {b.imag}")
        if b.real < 0:
            if b.real < -1e-3:
                return (False, f"Phase convention violated: b should be positive real, got {b.real}")
        return (True, "Phase convention: b is real (and positive if applicable)")

    def test_modeling_psi_components_consistent(self):
        """
        Check that psi = [a, b] (i.e., psi[0]=a, psi[1]=b)
        """
        if self.psi is None or self.a is None or self.b is None:
            return (True, "SKIP: insufficient data")
        try:
            psi0 = complex(self.psi[0])
            psi1 = complex(self.psi[1])
            a = complex(self.a)
            b = complex(self.b)
        except (TypeError, ValueError, IndexError):
            return (False, "Cannot extract components")
        if not (math.isclose(psi0.real, a.real, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(psi0.imag, a.imag, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(psi1.real, b.real, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(psi1.imag, b.imag, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, f"psi = ({psi0},{psi1}) does not match a={a}, b={b}")
        return (True, "psi components consistent with a and b")

    # ------------------------------------------------------------
    # 4. Monotonicity test: perturb input and compare to baseline
    # ------------------------------------------------------------
    def test_monotonicity_a_vs_b_linear_scaling(self):
        """
        Check that a scales linearly with b (i.e., a = i*b). 
        Perturb b by 10% and verify that a changes by the same factor.
        """
        if self.a is None or self.b is None:
            return (True, "SKIP: a or b not available")
        try:
            a = complex(self.a)
            b = complex(self.b)
        except (TypeError, ValueError):
            return (False, "a or b not complex")
        if abs(b) < 1e-15:
            return (True, "SKIP: b is essentially zero, cannot perform meaningful perturbation")
        factor = 1.1  # 10% increase
        b_pert = factor * b
        a_pert_expected = 1j * b_pert  # from the physical relation
        # Compare fractional changes (should be equal)
        fraction_a = (a_pert_expected - a) / a
        fraction_b = factor - 1  # 0.1
        if not (math.isclose(fraction_a.real, fraction_b, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(fraction_a.imag, 0.0, abs_tol=1e-2)):
            return (False, f"Monotonicity violated: fractional change in a ({fraction_a}) should equal fractional change in b ({fraction_b})")
        return (True, "Monotonicity: a scaled by same factor as b, consistent with linear relationship")

    # ------------------------------------------------------------
    # 5. Answer sanity (weak only, combined checks into one method)
    # ------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: check that psi is finite and has at least one nonzero component.
        No isclose, no re-derivation of exact values.
        """
        if self.psi is None:
            return (True, "SKIP: psi not available")
        try:
            psi0 = complex(self.psi[0])
            psi1 = complex(self.psi[1])
        except (TypeError, ValueError, IndexError):
            return (False, "Cannot interpret psi components")
        # Check finite
        for c in [psi0, psi1]:
            if math.isinf(c.real) or math.isinf(c.imag) or math.isnan(c.real) or math.isnan(c.imag):
                return (False, f"Component {c} is infinite or NaN")
        # Check at least one nonzero
        if abs(psi0) < 1e-10 and abs(psi1) < 1e-10:
            return (False, "Both components are essentially zero")
        return (True, "Psi is finite and has non-zero components")

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound methods starting with "test_"
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