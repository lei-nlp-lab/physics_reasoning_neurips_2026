import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for problem: plane EM wave and charged particle.
    Expected results keys: 'A_x', 'p_perp', and optionally 'A_y', 'A_z' or 'A' vector.
    """
    def __init__(self, solution_instance):
        # Safely extract givens
        self.e = getattr(solution_instance, 'e', None)
        self.m = getattr(solution_instance, 'm', None)

        # Safely extract results from dict or attribute
        self.A_x = None
        self.p_perp = None
        self.A_y = None
        self.A_z = None
        self.A = None

        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
            self.A_x = res.get('A_x', None)
            self.p_perp = res.get('p_perp', None)
            self.A_y = res.get('A_y', None)
            self.A_z = res.get('A_z', None)
            self.A = res.get('A', None)  # could be vector
        else:
            # Fallback: try attributes directly
            self.A_x = getattr(solution_instance, 'A_x', None)
            self.p_perp = getattr(solution_instance, 'p_perp', None)

        # If p_perp is a vector (list/tuple), assume first component is p_y, second p_z
        if isinstance(self.p_perp, (list, tuple)) and len(self.p_perp) == 2:
            self.p_perp_y = self.p_perp[0]
            self.p_perp_z = self.p_perp[1]
        else:
            self.p_perp_y = None
            self.p_perp_z = None

        # Also parse A components if given as vector
        if self.A is not None and isinstance(self.A, (list, tuple)) and len(self.A) >= 3:
            # A = (A_x, A_y, A_z) assumed
            if self.A_x is None:
                self.A_x = self.A[0]
            if self.A_y is None:
                self.A_y = self.A[1]
            if self.A_z is None:
                self.A_z = self.A[2]

        # Store solution_instance for optional access to explanation_steps
        self.solution_instance = solution_instance

    # ------------------------------------------------------------------
    # Interface / sanity checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that essential result keys exist."""
        if self.A_x is None:
            return (False, "Missing key: A_x")
        if self.p_perp is None:
            return (False, "Missing key: p_perp")
        return (True, "Required keys A_x and p_perp present")

    def test_interface_givens_present(self):
        """Check that charge e is present (mass m is optional for the derived relations)."""
        if self.e is None:
            return (False, "Missing given: e (charge)")
        return (True, "Charge e is present")

    # ------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ------------------------------------------------------------------
    def test_invariant_Ax_zero(self):
        """From gauge condition ∇·A=0 and vanishing at infinity, A_x must be exactly zero."""
        if self.A_x is None:
            return (False, "Cannot check A_x: value not available")

        try:
            val = float(self.A_x)
            if math.isclose(val, 0.0, rel_tol=1e-10, abs_tol=1e-10):
                return (True, f"A_x = {val} (effectively zero)")
            else:
                return (False, f"A_x = {val}, expected 0")
        except (TypeError, ValueError):
            # Possibly symbolic expression
            try:
                zero_expr = sp.simplify(self.A_x)
                if zero_expr == 0 or zero_expr.is_zero:
                    return (True, "A_x simplifies to zero symbolically")
                else:
                    return (False, f"A_x symbolic expression {zero_expr} not zero")
            except Exception:
                return (False, "Could not evaluate A_x numerically or symbolically")

    def test_invariant_p_perp_relation(self):
        """Check p_perp = -e * A_perp (componentwise if possible)."""
        if self.p_perp is None:
            return (False, "Cannot check p_perp relation: p_perp missing")
        if self.e is None:
            return (False, "Cannot check p_perp relation: e missing")

        # Obtain A_perp components
        A_y = self.A_y
        A_z = self.A_z
        if A_y is None or A_z is None:
            return (False, "Cannot check p_perp relation: transverse components of A not available")

        # Check p_y = -e * A_y
        if self.p_perp_y is not None:
            try:
                p_y = float(self.p_perp_y)
                e = float(self.e)
                A_y_val = float(A_y)
                if math.isclose(p_y, -e * A_y_val, rel_tol=1e-2, abs_tol=1e-2):
                    pass  # component ok
                else:
                    return (False, f"p_y = {p_y} does not equal -e*A_y = {-e*A_y_val}")
            except (TypeError, ValueError):
                return (False, "Cannot convert p_y / e / A_y to float")
        else:
            return (False, "p_perp components not parsed (expected list of two numbers)")

        # Check p_z = -e * A_z
        if self.p_perp_z is not None:
            try:
                p_z = float(self.p_perp_z)
                e = float(self.e)
                A_z_val = float(A_z)
                if math.isclose(p_z, -e * A_z_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "p_perp = -e * A_perp verified")
                else:
                    return (False, f"p_z = {p_z} does not equal -e*A_z = {-e*A_z_val}")
            except (TypeError, ValueError):
                return (False, "Cannot convert p_z / e / A_z to float")
        else:
            return (False, "p_perp components not parsed")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_transversality(self):
        """Check wave transversality: A_x=0 and A_perp components exist (non-zero not required)."""
        if self.A_x is None:
            return (False, "Cannot check transversality: A_x missing")
        try:
            Ax = float(self.A_x)
            if not math.isclose(Ax, 0.0, rel_tol=1e-6, abs_tol=1e-6):
                return (False, f"A_x = {Ax} should be zero (transverse wave)")
        except (TypeError, ValueError):
            return (False, "Cannot convert A_x to float")

        # Check that we have some perpendicular information
        if self.A_y is None and self.A_z is None:
            return (False, "Cannot verify transverse components: A_y and A_z both missing")
        return (True, "A_x zero, transverse components present")

    def test_modeling_orientation(self):
        """Check that p_perp is anti-parallel to A_perp (p_perp = -e A_perp implies opposite direction for positive e)."""
        if self.e is None:
            return (False, "Missing e for orientation check")
        if self.p_perp_y is None or self.p_perp_z is None:
            return (False, "Cannot check orientation: p_perp components not available")
        if self.A_y is None or self.A_z is None:
            return (False, "Cannot check orientation: A_perp components not available")

        try:
            e = float(self.e)
            p_y = float(self.p_perp_y)
            p_z = float(self.p_perp_z)
            A_y_val = float(self.A_y)
            A_z_val = float(self.A_z)

            # Check that vectors are (anti)parallel: p_perp should be proportional to -A_perp.
            # p_y / A_y_val should equal p_z / A_z_val = -e (within tolerance).
            # Handle possible zero components.
            if abs(A_y_val) > 1e-12 and abs(A_z_val) > 1e-12:
                ratio_y = p_y / A_y_val
                ratio_z = p_z / A_z_val
                if math.isclose(ratio_y, ratio_z, rel_tol=1e-2, abs_tol=1e-2) and math.isclose(ratio_y, -e, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "p_perp and -A_perp are parallel with correct magnitude")
                else:
                    return (False, f"Expected p_perp = -e A_perp, but ratios differ: {ratio_y}, {ratio_z}, -e = {-e}")
            elif abs(A_y_val) > 1e-12:
                # only y component, check direction
                expected_p_y = -e * A_y_val
                if math.isclose(p_y, expected_p_y, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "y-component relation holds, z-component zero")
                else:
                    return (False, f"p_y mismatch: {p_y} vs expected {expected_p_y}")
            elif abs(A_z_val) > 1e-12:
                expected_p_z = -e * A_z_val
                if math.isclose(p_z, expected_p_z, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "z-component relation holds, y-component zero")
                else:
                    return (False, f"p_z mismatch: {p_z} vs expected {expected_p_z}")
            else:
                return (False, "Both A_perp components zero; relation trivially satisfied but no orientation check")
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float for orientation check")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_scaling(self):
        """
        Verify that p_perp scales linearly with A_perp.
        Using the existing data as baseline, we perturb A_perp by a factor λ=2.0
        and check that p_perp would also scale by λ (since p_perp = -e A_perp).
        """
        if self.e is None:
            return (False, "Cannot run monotonicity: e missing")
        if self.p_perp_y is None or self.p_perp_z is None:
            return (False, "Cannot run monotonicity: p_perp components missing")
        if self.A_y is None or self.A_z is None:
            return (False, "Cannot run monotonicity: A_perp components missing")

        try:
            e = float(self.e)
            p_y = float(self.p_perp_y)
            p_z = float(self.p_perp_z)
            A_y_val = float(self.A_y)
            A_z_val = float(self.A_z)

            # Baseline: check consistency (should be true if previous tests pass)
            # We will trust that baseline holds.
            # Now perturb A_perp by factor λ=2.0
            lam = 2.0
            A_y_new = lam * A_y_val
            A_z_new = lam * A_z_val
            # Expected p_perp from linear relation
            p_y_expected = -e * A_y_new   # = lam * (-e*A_y) = lam * p_y
            p_z_expected = -e * A_z_new   # = lam * p_z

            # Since we don't have actual perturbed results, we check that the predicted formula
            # yields p_y_expected, p_z_expected from the baseline relation.
            # This is effectively verifying the linear relation itself.
            # But we can compare: lam * p_y should equal -e * (lam * A_y_val)
            if math.isclose(p_y_expected, lam * p_y, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(p_z_expected, lam * p_z, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p_perp scales linearly with A_perp (λ={lam}): verified")
            else:
                return (False, f"Scaling check failed: p_new expected {p_y_expected, p_z_expected}, got λ*p = {lam*p_y, lam*p_z}")
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float for monotonicity")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Basic sanity: A_x is zero; p_perp and A_perp are finite; e non-zero."""
        if self.A_x is None:
            return (False, "A_x not provided")
        if self.p_perp is None:
            return (False, "p_perp not provided")

        # Check A_x finite and close to 0 (very loose)
        try:
            Ax = float(self.A_x)
            if not math.isfinite(Ax):
                return (False, "A_x is not finite")
            if abs(Ax) > 1e-6:
                return (False, f"A_x = {Ax} is not near zero")
        except (TypeError, ValueError):
            return (False, "Cannot convert A_x to float")

        # Check p_perp finite (components if available)
        if self.p_perp_y is not None and self.p_perp_z is not None:
            try:
                py = float(self.p_perp_y)
                pz = float(self.p_perp_z)
                if not (math.isfinite(py) and math.isfinite(pz)):
                    return (False, "p_perp components are not finite")
            except (TypeError, ValueError):
                return (False, "Cannot convert p_perp components to float")
        else:
            # Try the whole p_perp as numeric
            try:
                p_val = float(self.p_perp)
                if not math.isfinite(p_val):
                    return (False, "p_perp is not finite")
            except (TypeError, ValueError):
                # Could be symbolic; not a failure
                pass

        # Check that e is non-zero to avoid division issues
        try:
            e_val = float(self.e)
            if e_val == 0.0:
                return (False, "Charge e is zero, which would make p_perp zero trivially but unphysical")
        except (TypeError, ValueError):
            return (False, "Cannot interpret e as number")

        return (True, "Basic sanity: A_x near zero, p_perp finite, e non-zero")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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