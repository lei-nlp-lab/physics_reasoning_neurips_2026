import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the spectral line coherence length problem.
    
    Expected keys in solution_instance:
      - lambda (numeric): central wavelength in nm
      - delta_lambda (numeric): linewidth in nm
      - results (dict): must contain 'L_c_cm' as the coherence length in cm
      - explanation_steps (list of strings, optional): reasoning steps
    Missing values are set to None.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.lambda_nm = getattr(solution_instance, 'lambda', None)
        self.delta_lambda_nm = getattr(solution_instance, 'delta_lambda', None)

        # Extract results safely; expect a dict
        results = getattr(solution_instance, 'results', None)
        if results and isinstance(results, dict):
            self.L_c_cm = results.get('L_c_cm', None)
        else:
            self.L_c_cm = None

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required inputs and results are present."""
        missing = []
        if self.lambda_nm is None:
            missing.append('lambda')
        if self.delta_lambda_nm is None:
            missing.append('delta_lambda')
        if self.L_c_cm is None:
            missing.append('L_c_cm')
        if missing:
            return (False, f"Missing required keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_L_c_cm_positive(self):
        """Coherence length must be positive."""
        if self.L_c_cm is None:
            return (False, "SKIP: L_c_cm not available")
        if self.L_c_cm <= 0:
            return (False, f"L_c_cm = {self.L_c_cm} is not positive")
        return (True, f"L_c_cm = {self.L_c_cm} > 0")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ----------------------------------------------------------------------
    def test_invariant_coherence_length_formula(self):
        """Check L_c ≈ λ² / Δλ with tolerance (quasi-monochromatic)."""
        if None in (self.lambda_nm, self.delta_lambda_nm, self.L_c_cm):
            return (False, "SKIP: missing lambda, delta_lambda or L_c_cm")

        try:
            lam = float(self.lambda_nm)
            dlam = float(self.delta_lambda_nm)
            L_c_in = float(self.L_c_cm)
        except (ValueError, TypeError):
            return (False, "Non-numeric values encountered")

        # Compute expected L_c in cm directly using nanometers:
        # λ in nm, Δλ in nm → λ²/Δλ in nm conversion to cm: 1 nm = 1e-7 cm
        # L_c (cm) = (λ²/Δλ) * 1e-7
        L_c_expected_cm = (lam ** 2 / dlam) * 1e-7

        # Use isclose for fuzzy matching (rel_tol 1e-2, abs_tol 1e-2)
        if math.isclose(L_c_in, L_c_expected_cm, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L_c = {L_c_in} cm matches expected {L_c_expected_cm} cm within tolerance")
        else:
            return (False, f"L_c = {L_c_in} cm differs from expected {L_c_expected_cm} cm")

    def test_invariant_quasimonochromatic_condition(self):
        """Δλ must be much smaller than λ (quasi-monochromatic)."""
        if None in (self.lambda_nm, self.delta_lambda_nm):
            return (False, "SKIP: lambda or delta_lambda missing")
        try:
            lam = float(self.lambda_nm)
            dlam = float(self.delta_lambda_nm)
        except (ValueError, TypeError):
            return (False, "Non-numeric values")

        if dlam <= 0:
            return (False, f"Δλ = {dlam} is not positive")
        ratio = lam / dlam
        if ratio < 1000:
            return (False, f"λ/Δλ = {ratio:.2f} is less than 1000, not clearly quasi-monochromatic")
        return (True, f"λ/Δλ = {ratio:.2f} > 1000 → quasi-monochromatic satisfied")

    # ----------------------------------------------------------------------
    # Modeling/Trace checks (at least two, no hardcoded target answers)
    # ----------------------------------------------------------------------
    def test_modeling_explanation_contains_coherence_formula(self):
        """
        Check that the explanation steps mention the coherence length formula
        or the concept of coherence length.
        This verifies the modeling path includes the core physics relation.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        if not isinstance(self.explanation_steps, list):
            return (False, "explanation_steps is not a list")
        lines = " ".join(str(step) for step in self.explanation_steps)
        found = ("λ²/Δλ" in lines or "lambda^2/delta_lambda" in lines.lower() or 
                 "coherence length" in lines.lower())
        if found:
            return (True, "Explanation contains coherence length formula/term")
        else:
            return (False, "Explanation does not contain the required coherence length formula")

    def test_modeling_explanation_visibility_condition(self):
        """
        Check that the explanation steps mention the visibility condition:
        path difference ≤ coherence length (or fraction thereof).
        This ensures the modeling connects coherence length to fringe visibility.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        if not isinstance(self.explanation_steps, list):
            return (False, "explanation_steps is not a list")
        lines = " ".join(str(step) for step in self.explanation_steps).lower()
        # Look for combined mention of path difference and coherence length
        has_path = "path difference" in lines or "optical path" in lines
        has_coherence = "coherence length" in lines or "l_c" in lines or "lc" in lines
        # Also check for the fraction "quarter" or "1/4"
        has_fraction = "quarter" in lines or "1/4" in lines or "0.25" in lines
        if has_path and has_coherence:
            return (True, "Explanation connects path difference and coherence length")
        elif has_coherence and has_fraction:
            return (True, "Explanation mentions fraction of coherence length (quarter)")
        else:
            return (False, "Explanation does not clearly link path difference and coherence length")

    # ----------------------------------------------------------------------
    # Monotonicity checks (re-derive using physics relation)
    # ----------------------------------------------------------------------
    def test_monotonicity_lambda_increase_increases_L_c(self):
        """
        If λ increases (while Δλ fixed), L_c should increase.
        Perturb λ by +1% and check L_c increases.
        """
        if None in (self.lambda_nm, self.delta_lambda_nm, self.L_c_cm):
            return (False, "SKIP: missing inputs")
        try:
            lam = float(self.lambda_nm)
            dlam = float(self.delta_lambda_nm)
        except (ValueError, TypeError):
            return (False, "Non-numeric values")

        # Baseline L_c (computed from formula to avoid trusting stored value)
        L_c_baseline = (lam ** 2 / dlam) * 1e-7

        # Perturb λ upward by 1%
        lam_pert = lam * 1.01
        L_c_pert = (lam_pert ** 2 / dlam) * 1e-7

        if L_c_pert > L_c_baseline:
            return (True, f"λ increased 1% → L_c increased from {L_c_baseline} to {L_c_pert}")
        else:
            return (False, f"λ increased but L_c did not increase")

    def test_monotonicity_delta_lambda_increase_decreases_L_c(self):
        """
        If Δλ increases (while λ fixed), L_c should decrease.
        Perturb Δλ by +1% and check L_c decreases.
        """
        if None in (self.lambda_nm, self.delta_lambda_nm, self.L_c_cm):
            return (False, "SKIP: missing inputs")
        try:
            lam = float(self.lambda_nm)
            dlam = float(self.delta_lambda_nm)
        except (ValueError, TypeError):
            return (False, "Non-numeric values")

        L_c_baseline = (lam ** 2 / dlam) * 1e-7

        # Perturb Δλ upward by 1%
        dlam_pert = dlam * 1.01
        L_c_pert = (lam ** 2 / dlam_pert) * 1e-7

        if L_c_pert < L_c_baseline:
            return (True, f"Δλ increased 1% → L_c decreased from {L_c_baseline} to {L_c_pert}")
        else:
            return (False, f"Δλ increased but L_c did not decrease")

    # ----------------------------------------------------------------------
    # Weak answer sanity (does not use math.isclose, does not re-derive)
    # ----------------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self):
        """
        Check that L_c_cm is of order ~1 cm (between 0.1 and 10 cm).
        This is a weak sanity check; exact value is not enforced.
        """
        if self.L_c_cm is None:
            return (False, "SKIP: L_c_cm not available")
        try:
            val = float(self.L_c_cm)
        except (ValueError, TypeError):
            return (False, "L_c_cm is not a valid number")
        if not (0.1 <= val <= 10):
            return (False, f"L_c = {val} cm is outside expected range (0.1–10 cm)")
        return (True, f"L_c = {val} cm within plausible order-of-magnitude")

    # ----------------------------------------------------------------------
    # Aggregate test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (success, logs)."""
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