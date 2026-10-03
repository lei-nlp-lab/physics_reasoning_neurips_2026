import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    A robust checker for the parabolic mirror diffraction problem.
    """
    def __init__(self, solution_instance):
        # Safe extraction of givens and results
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract parameters with None defaults
        self.D = self.givens.get('D', None)
        self.f = self.givens.get('f', None)
        self.lambda_val = self.givens.get('lambda', None)  # may be in cm or nm
        self.alpha = self.givens.get('alpha', None)  # off-axis angle (if present)

        # Result keys
        self.r_airydisk = self.results.get('r_airydisk', None)

        # Convert lambda to cm if given in nm (typical value ~550 nm = 5.5e-5 cm)
        if self.lambda_val is not None and self.lambda_val > 1e-4:
            # likely in nm, convert to cm (1 nm = 1e-7 cm)
            self.lambda_cm = self.lambda_val * 1e-7
        else:
            self.lambda_cm = self.lambda_val  # assume already cm

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

    # ---------- Interface Tests ----------
    def test_interface_airydisk_exists(self):
        """Check that r_airydisk is present in results and is numeric."""
        if self.r_airydisk is None:
            return (False, "r_airydisk not found in results")
        if not isinstance(self.r_airydisk, (int, float)):
            return (False, f"r_airydisk is not numeric: {type(self.r_airydisk)}")
        return (True, "r_airydisk exists and is numeric")

    def test_interface_givens_available(self):
        """Check that required givens are present (D, f, lambda)."""
        missing = []
        if self.D is None:
            missing.append("D")
        if self.f is None:
            missing.append("f")
        if self.lambda_val is None:
            missing.append("lambda")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ---------- Invariant Tests ----------
    def test_invariant_rayleigh_criterion(self):
        """Check that r_airydisk satisfies r = 1.22 * lambda * f / D within 5% tolerance."""
        if any(x is None for x in [self.r_airydisk, self.D, self.f, self.lambda_cm]):
            return (False, "SKIP: missing data for Rayleigh check")
        try:
            expected = 1.22 * self.lambda_cm * self.f / self.D
        except Exception as e:
            return (False, f"Cannot compute expected radius: {e}")
        if math.isclose(self.r_airydisk, expected, rel_tol=0.05, abs_tol=1e-6):
            return (True, f"Rayleigh criterion satisfied: r={self.r_airydisk:.6e}, expected={expected:.6e}")
        else:
            return (False, f"Rayleigh mismatch: r={self.r_airydisk:.6e}, expected={expected:.6e}")

    def test_invariant_positive_radius(self):
        """Check that r_airydisk is positive."""
        if self.r_airydisk is None:
            return (False, "SKIP: r_airydisk missing")
        if self.r_airydisk > 0:
            return (True, "Radius is positive")
        else:
            return (False, f"Radius non-positive: {self.r_airydisk}")

    def test_invariant_offaxis_diffraction_scaling(self):
        """If off-axis angle alpha is given, check that results include a note about cosθ scaling."""
        if self.alpha is None:
            return (True, "No off-axis angle, trivial pass")
        # Check that explanation mentions cosθ or coma (qualitative check)
        if self.explanation_steps is not None:
            if 'cos' in self.explanation_steps or 'coma' in self.explanation_steps:
                return (True, "Off-axis scaling acknowledged")
            else:
                return (False, "Off-axis scaling not mentioned in explanation")
        return (True, "No explanation steps to check")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_rayleigh_formula_consistency(self):
        """Use Sympy to verify the symbolic formula: r = 1.22*lambda*f/D."""
        if any(x is None for x in [self.r_airydisk, self.D, self.f, self.lambda_cm]):
            return (False, "SKIP: missing data for symbolic check")
        # symbolic variables
        D_sym, f_sym, lam_sym = sp.symbols('D f lam')
        expr = 1.22 * lam_sym * f_sym / D_sym
        # numerical substitution
        try:
            numeric_expr = expr.subs({D_sym: self.D, f_sym: self.f, lam_sym: self.lambda_cm})
            expected_float = float(numeric_expr.evalf())
        except Exception as e:
            return (False, f"Symbolic evaluation failed: {e}")
        if math.isclose(self.r_airydisk, expected_float, rel_tol=0.05, abs_tol=1e-6):
            return (True, f"Symbolic formula matches: {self.r_airydisk:.6e} ≈ {expected_float:.6e}")
        else:
            return (False, f"Symbolic mismatch: {self.r_airydisk:.6e} != {expected_float:.6e}")

    def test_modeling_airydisk_scale_consistency(self):
        """
        Verify that the reported Airy disk radius is consistent with the scale 
        derived from the Rayleigh formula and the given parameters.
        This re-derives the expected radius from givens and checks that the 
        reported value lies within a factor of 2 of that expected value.
        """
        if any(x is None for x in [self.r_airydisk, self.D, self.f, self.lambda_cm]):
            return (False, "SKIP: missing data for scale consistency")
        try:
            expected = 1.22 * self.lambda_cm * self.f / self.D
        except Exception as e:
            return (False, f"Cannot compute expected radius: {e}")
        # Accept if reported radius is within factor 2 (0.5 to 2.0)
        if expected == 0:
            return (False, "Expected radius is zero, invalid")
        ratio = self.r_airydisk / expected
        if 0.5 <= ratio <= 2.0:
            return (True, f"Scale consistent: ratio = {ratio:.3f} (within factor 2)")
        else:
            return (False, f"Scale inconsistent: ratio = {ratio:.3f}, expected ~ {expected:.6e}")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_diameter_increase_reduces_radius(self):
        """
        Check monotonicity: if the aperture diameter is increased (by 10%),
        the Airy disk radius should decrease.
        Uses the Rayleigh formula to compute baseline and perturbed radii.
        """
        if any(x is None for x in [self.D, self.f, self.lambda_cm]):
            return (False, "SKIP: missing givens for monotonicity check")
        # Baseline radius using given formula
        try:
            r_baseline = 1.22 * self.lambda_cm * self.f / self.D
        except Exception as e:
            return (False, f"Cannot compute baseline radius: {e}")
        # Perturb: increase D by 10%
        D_pert = self.D * 1.1
        try:
            r_pert = 1.22 * self.lambda_cm * self.f / D_pert
        except Exception as e:
            return (False, f"Cannot compute perturbed radius: {e}")
        # Monotonicity: increasing D reduces radius
        if r_pert < r_baseline:
            return (True, f"Monotonicity holds: D increased by 10% -> radius decreased from {r_baseline:.6e} to {r_pert:.6e}")
        else:
            return (False, f"Monotonicity violated: D increased but radius did not decrease (baseline {r_baseline:.6e}, perturbed {r_pert:.6e})")

    # ---------- Answer Sanity (weak) ----------
    def test_answer_sanity_broad_range(self):
        """Check that r_airydisk is within a very broad physical range (10^-6 to 10 cm)."""
        if self.r_airydisk is None:
            return (False, "SKIP: r_airydisk missing")
        if 1e-6 < self.r_airydisk < 10.0:
            return (True, f"Radius {self.r_airydisk:.6e} cm within plausible range")
        else:
            return (False, f"Radius {self.r_airydisk:.6e} cm outside plausible range [1e-6, 10]")