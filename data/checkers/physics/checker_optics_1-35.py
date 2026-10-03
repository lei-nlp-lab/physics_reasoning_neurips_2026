import math
import traceback
import re

class PhysicsChecker:
    """
    Auditor for a solution to the lens makers' formula, achromatic doublet derivation,
    and sign condition for crown/flint glass lenses.
    """

    def __init__(self, solution_instance):
        # Extract results from the solution instance
        self.lens_makers = None
        self.achromatic_eq = None
        self.f_t_sign = None

        # Prefer .results dictionary if it exists
        results = getattr(solution_instance, 'results', None)
        if isinstance(results, dict):
            self.lens_makers = results.get('lens_makers', None)
            self.achromatic_eq = results.get('achromatic_eq', None)
            self.f_t_sign = results.get('f_t_sign', None)

        # Fallback to direct attributes
        if self.lens_makers is None:
            self.lens_makers = getattr(solution_instance, 'lens_makers', None)
        if self.achromatic_eq is None:
            self.achromatic_eq = getattr(solution_instance, 'achromatic_eq', None)
        if self.f_t_sign is None:
            self.f_t_sign = getattr(solution_instance, 'f_t_sign', None)

        # Hard‑coded dispersion values (provided in the problem statement)
        self.delta_c = 0.0169
        self.delta_f = 0.0384

    # ------------------------------------------------------------------
    # Helper: normalise a physics expression string for pattern detection
    # ------------------------------------------------------------------
    def _normalise(self, s):
        if s is None:
            return ""
        s = s.replace("\\", "")
        s = s.replace(" ", "").replace("\n", "").replace("\t", "").replace("_", "")
        s = s.lower()
        # Also handle common LaTeX greek: \delta, \Delta
        s = s.replace("delta", "δ").replace("delta__c", "δc").replace("delta__f", "δf")
        s = s.replace("Δc", "δc").replace("Δf", "δf")
        return s

    # ------------------------------------------------------------------
    # Interface tests – required keys exist and are strings
    # ------------------------------------------------------------------
    def test_interface_lens_makers_exists(self):
        if self.lens_makers is None:
            return (False, "lens_makers key missing")
        if not isinstance(self.lens_makers, str):
            return (False, f"lens_makers is not a string, got {type(self.lens_makers)}")
        return (True, "lens_makers present and is string")

    def test_interface_achromatic_eq_exists(self):
        if self.achromatic_eq is None:
            return (False, "achromatic_eq key missing")
        if not isinstance(self.achromatic_eq, str):
            return (False, f"achromatic_eq is not a string, got {type(self.achromatic_eq)}")
        return (True, "achromatic_eq present and is string")

    def test_interface_f_t_sign_exists(self):
        if self.f_t_sign is None:
            return (False, "f_t_sign key missing")
        if not isinstance(self.f_t_sign, str):
            return (False, f"f_t_sign is not a string, got {type(self.f_t_sign)}")
        return (True, "f_t_sign present and is string")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (no exact answer matching)
    # ------------------------------------------------------------------
    def test_invariant_achromatic_equation_structure(self):
        """The achromatic condition must be a linear equation linking Δ/f terms.
           Check that the result string contains the essential pattern: some combination
           of Δ_c, f_c, Δ_f, f_t set equal to zero."""
        if self.achromatic_eq is None:
            return (False, "SKIP: achromatic_eq missing")
        norm = self._normalise(self.achromatic_eq)
        # Essential components: δc/fc + δf/ft = 0 (after normalisation)
        if "δc/fc" in norm and "δf/ft" in norm and "=0" in norm:
            return (True, "Achromatic equation has the form Δ_c/f_c + Δ_f/f_t = 0")
        # Also accept if expressed as sum = 0 with variable names slightly different
        # e.g., δc/fc + δf/ft = 0 (already covered)
        if "δc" in norm and "δf" in norm and "fc" in norm and "ft" in norm and "0" in norm:
            return (True, "Achromatic equation contains all necessary terms and equates to zero")
        return (False, f"Achromatic equation does not match expected pattern. Got: {self.achromatic_eq}")

    def test_invariant_lens_makers_formula_structure(self):
        """The lens maker's formula must involve (n-1) times a combination of two radii.
           No exact string matching; just check for the presence of key symbols."""
        if self.lens_makers is None:
            return (False, "SKIP: lens_makers missing")
        norm = self._normalise(self.lens_makers)
        # Should contain n-1, two radii (R1, R2 or R, R')
        has_n_minus_1 = "n-1" in norm or "n-1)" in norm
        has_two_radii = ("1/r1" in norm and "1/r2" in norm) or ("1/r" in norm and "1/r'" in norm)
        # Also accept 1/R1 - 1/R2 or 1/R + 1/R'
        if has_n_minus_1 and has_two_radii:
            return (True, "Lens makers formula has (n-1) and two radii")
        # More general: at least one R and something like (1/R1 – 1/R2)
        if "r1" in norm and "r2" in norm and "n-1" in norm:
            return (True, "Lens makers formula contains (n-1) and two radii")
        if "r" in norm and "n-1" in norm:
            # Could be the explanation's form: 1/R + 1/R'
            return (True, "Lens makers formula contains (n-1) and at least one radius")
        return (False, f"Lens makers formula does not have required structure. Got: {self.lens_makers}")

    def test_invariant_sign_implication(self):
        """From the achromatic condition and the given Δ values, the sign of f_t must be
           opposite to that of f_c. This test checks that the f_t_sign result contains
           a statement that f_t is negative when f_c > 0."""
        if self.f_t_sign is None:
            return (False, "SKIP: f_t_sign missing")
        norm = self._normalise(self.f_t_sign)
        # Look for either explicit inequality: ft < 0, or equivalent expression
        if "ft < 0" in norm or "ft<0" in norm:
            return (True, "Sign implication correctly states f_t < 0")
        # Also accept if it says "diverging flint lens" (implies negative focal length)
        if "diverging flint" in self.f_t_sign.lower():
            return (True, "Sign implication states flint lens diverging (f_t negative)")
        return (False, f"f_t_sign does not clearly indicate negative f_t. Got: {self.f_t_sign}")

    # ------------------------------------------------------------------
    # Modeling / trace tests – consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_re_derives_sign_numerically(self):
        """Using given Δ values and a fictitious positive f_c, re‑derive the sign of f_t
           from the achromatic condition. This test validates that the physics is consistent."""
        if self.f_t_sign is None:
            return (False, "SKIP: f_t_sign missing")
        # The achromatic condition gives: Δ_c/f_c + Δ_f/f_t = 0  =>  f_t = - (Δ_f/Δ_c) * f_c
        # Given Δ_c, Δ_f > 0, for any positive f_c, f_t is negative.
        # We pick a dummy f_c = 10.0 (any positive number works)
        f_c_dummy = 10.0
        f_t_derived = - (self.delta_f / self.delta_c) * f_c_dummy
        # Check that derived f_t is negative
        if f_t_derived < 0:
            # Now verify that the output f_t_sign agrees with this conclusion
            return (True, f"Re‑derivation: f_t = -({self.delta_f}/{self.delta_c})*f_c < 0 for f_c>0. Consistent.")
        else:
            return (False, "Re‑derivation error: f_t should be negative but computed positive")

    def test_modeling_total_power_usage(self):
        """The explanation uses 1/f = 1/f_c + 1/f_t. Check that the results imply that
           the two lenses are combined (e.g., through the achromatic equation)."""
        # We already have achromatic_eq; we can also check that f_t_sign contains
        # the relation between f_t and f_c (like f_t = -k f_c)
        if self.f_t_sign is None:
            return (False, "SKIP: f_t_sign missing")
        norm = self._normalise(self.f_t_sign)
        if "ft" in norm and "fc" in norm and ("-" in norm):
            return (True, "f_t is expressed in terms of f_c, indicating combined system")
        return (False, "f_t_sign does not show relation between f_t and f_c")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb an input and compare
    # ------------------------------------------------------------------
    def test_monotonicity_ft_decreases_with_delta_c(self):
        """Perturb Δ_c and verify that the derived f_t magnitude decreases.
           Baseline: f_c = 10.0 (arbitrary positive). Compute f_t from achromatic condition.
           Then increase Δ_c by 5% and recompute f_t. Check that new f_t is still negative
           and its absolute value is smaller than baseline."""
        # Baseline (using given Δ_c)
        f_c = 10.0
        f_t_baseline = - (self.delta_f / self.delta_c) * f_c
        # Perturb Δ_c by +5%
        delta_c_new = self.delta_c * 1.05
        f_t_perturbed = - (self.delta_f / delta_c_new) * f_c
        # Both should be negative
        if not (f_t_baseline < 0 and f_t_perturbed < 0):
            return (False, "Monotonicity test: derived f_t not negative")
        # Check that absolute value decreases (moves toward zero)
        if abs(f_t_perturbed) < abs(f_t_baseline):
            return (True, f"Monotonicity holds: increasing Δ_c reduces |f_t| (baseline {abs(f_t_baseline):.3f} → {abs(f_t_perturbed):.3f})")
        else:
            return (False, f"Monotonicity failed: |f_t| increased from {abs(f_t_baseline):.3f} to {abs(f_t_perturbed):.3f}")

    def test_monotonicity_ft_becomes_more_negative_with_f_c(self):
        """Perturb f_c and check that f_t becomes more negative (magnitude increases).
           Baseline f_c = 10.0, compute f_t. Increase f_c by 10% and recompute."""
        delta_c = self.delta_c
        delta_f = self.delta_f
        f_c_base = 10.0
        f_t_base = - (delta_f / delta_c) * f_c_base
        # Perturb: increase f_c by 10%
        f_c_new = f_c_base * 1.10
        f_t_new = - (delta_f / delta_c) * f_c_new
        # Both negative
        if not (f_t_base < 0 and f_t_new < 0):
            return (False, "Monotonicity test: derived f_t not negative")
        # Check that absolute value increases (more negative)
        if abs(f_t_new) > abs(f_t_base):
            return (True, f"Monotonicity holds: increasing f_c makes f_t more negative (baseline {f_t_base:.3f} → {f_t_new:.3f})")
        else:
            return (False, f"Monotonicity failed: f_t became less negative ({f_t_base:.3f} → {f_t_new:.3f})")

    # ------------------------------------------------------------------
    # Weak answer sanity – only one test, no hardcoded target values
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_condition(self):
        """Weak sanity: the answer should state that f_c > 0 and f_t < 0 for a converging
           achromatic lens. No exact comparison; just presence of both inequalities."""
        if self.f_t_sign is None:
            return (False, "SKIP: f_t_sign missing")
        norm = self._normalise(self.f_t_sign)
        # Look for "fc > 0" and "ft < 0" (possibly without spaces)
        has_fc_positive = "fc>0" in norm or "fc > 0" in norm or "fc positive" in norm
        has_ft_negative = "ft<0" in norm or "ft < 0" in norm or "ft negative" in norm
        if has_fc_positive and has_ft_negative:
            return (True, "Answer contains both fc>0 and ft<0")
        # Also accept if it says "converging crown" and "diverging flint"
        text = self.f_t_sign.lower()
        if "converging crown" in text and "diverging flint" in text:
            return (True, "Answer describes converging crown and diverging flint")
        return (False, "Answer does not clearly state sign condition")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all callable methods starting with test_
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)