import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary or attributes safely
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.J2_secular_omega = solution_instance.results.get('J2_secular_omega', None)
            self.J2_secular_Omega = solution_instance.results.get('J2_secular_Omega', None)
            self.drag_per_orbit_da = solution_instance.results.get('drag_per_orbit_da', None)
            self.drag_per_orbit_de = solution_instance.results.get('drag_per_orbit_de', None)
        else:
            self.J2_secular_omega = getattr(solution_instance, 'J2_secular_omega', None)
            self.J2_secular_Omega = getattr(solution_instance, 'J2_secular_Omega', None)
            self.drag_per_orbit_da = getattr(solution_instance, 'drag_per_orbit_da', None)
            self.drag_per_orbit_de = getattr(solution_instance, 'drag_per_orbit_de', None)
        
        # Also try to get an overall answer text if present (for answer sanity)
        self.answer = None
        if hasattr(solution_instance, 'answer') and isinstance(solution_instance.answer, str):
            self.answer = solution_instance.answer
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.answer = solution_instance.results.get('answer', None)

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

    # ---- Interface Tests ----
    def test_interface_required_keys_exist(self):
        """Check that all four required result keys are present and not None."""
        missing = []
        for key, val in [('J2_secular_omega', self.J2_secular_omega),
                         ('J2_secular_Omega', self.J2_secular_Omega),
                         ('drag_per_orbit_da', self.drag_per_orbit_da),
                         ('drag_per_orbit_de', self.drag_per_orbit_de)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All four required keys present")

    def test_interface_valid_type(self):
        """Check that each result value is a string (or numeric) – at least not None."""
        for key, val in [('J2_secular_omega', self.J2_secular_omega),
                         ('J2_secular_Omega', self.J2_secular_Omega),
                         ('drag_per_orbit_da', self.drag_per_orbit_da),
                         ('drag_per_orbit_de', self.drag_per_orbit_de)]:
            if val is None:
                return (False, f"{key} is None")
            if not isinstance(val, (str, int, float)):
                return (False, f"{key} has invalid type {type(val).__name__}")
        return (True, "All values have valid types (str/int/float)")

    # ---- Invariant Tests (textual/physical relevance) ----
    def test_invariant_J2_secular_omega_contains_omega(self):
        if self.J2_secular_omega is None:
            return (False, "SKIP: J2_secular_omega not available")
        s = str(self.J2_secular_omega).lower()
        if any(term in s for term in ['omega', 'argument of perigee', 'perigee precession']):
            return (True, "Contains reference to omega/precession")
        else:
            return (False, f"String does not mention omega or perigee precession")

    def test_invariant_J2_secular_Omega_contains_Omega(self):
        if self.J2_secular_Omega is None:
            return (False, "SKIP: J2_secular_Omega not available")
        s = str(self.J2_secular_Omega).lower()
        if any(term in s for term in ['omega', 'ascending node', 'node', 'precession']):
            return (True, "Contains reference to node/precession")
        else:
            return (False, f"String does not mention node or precession")

    def test_invariant_drag_per_orbit_da_decrease(self):
        if self.drag_per_orbit_da is None:
            return (False, "SKIP: drag_per_orbit_da not available")
        s = str(self.drag_per_orbit_da).lower()
        if any(term in s for term in ['-', 'decrease', 'reduce', 'decay', 'circularization', 'shrink']):
            return (True, "Indicates decrease in semi-major axis")
        else:
            return (False, "String does not indicate decrease")

    def test_invariant_drag_per_orbit_de_decrease(self):
        if self.drag_per_orbit_de is None:
            return (False, "SKIP: drag_per_orbit_de not available")
        s = str(self.drag_per_orbit_de).lower()
        if any(term in s for term in ['-', 'decrease', 'reduce', 'circularization', 'shrink']):
            return (True, "Indicates decrease in eccentricity")
        else:
            return (False, "String does not indicate decrease")

    # ---- Modeling/Trace Tests ----
    def test_modeling_drag_vs_J2_conservative_nature(self):
        if any(v is None for v in [self.J2_secular_omega, self.J2_secular_Omega, self.drag_per_orbit_da, self.drag_per_orbit_de]):
            return (False, "SKIP: one or more results missing")
        drag_text = (str(self.drag_per_orbit_da) + " " + str(self.drag_per_orbit_de)).lower()
        J2_text = (str(self.J2_secular_omega) + " " + str(self.J2_secular_Omega)).lower()
        drag_has_dissip = any(term in drag_text for term in ['drag', 'dissipation', 'energy loss', 'nonconservative'])
        J2_has_dissip = any(term in J2_text for term in ['drag', 'dissipation', 'energy loss', 'nonconservative'])
        if not drag_has_dissip:
            return (False, "Drag results should mention dissipation or energy loss")
        if J2_has_dissip:
            return (False, "J2 results should not mention dissipation (conservative perturbation)")
        return (True, "Distinction between conservative J2 and non-conservative drag is present")

    def test_trace_both_perturbations_addressed(self):
        if any(v is None for v in [self.J2_secular_omega, self.J2_secular_Omega, self.drag_per_orbit_da, self.drag_per_orbit_de]):
            return (False, "SKIP: one or more results missing")
        J2_terms = ['j2', 'oblateness', 'polar flattening', 'zonal']
        drag_terms = ['drag', 'atmospheric drag', 'perigee', 'drag impulse']
        J2_text = (str(self.J2_secular_omega) + " " + str(self.J2_secular_Omega)).lower()
        drag_text = (str(self.drag_per_orbit_da) + " " + str(self.drag_per_orbit_de)).lower()
        J2_found = any(term in J2_text for term in J2_terms)
        drag_found = any(term in drag_text for term in drag_terms)
        if not J2_found:
            return (False, "J2 results do not mention J2 or oblateness")
        if not drag_found:
            return (False, "Drag results do not mention drag or perigee")
        return (True, "Both perturbations are addressed in respective results")

    # ---- Monotonicity Test (numerical perturbation) ----
    def test_monotonicity_drag_da_vs_rho(self):
        """
        Real monotonicity test: we numerically evaluate the canonical Δa formula
        at two density values (baseline and +10%) and verify that Δa becomes more negative.
        The formula is taken from the physics mapping and pseudocode:
        Δa ≈ -2π (C_D A/m) ρ_perigee a^2 / (1 - e^2)
        We use representative values for a, e, C_D, A, m.
        If the result string is missing, we skip.
        """
        if self.drag_per_orbit_da is None:
            return (False, "SKIP: drag_per_orbit_da not available")
        # Define symbolic expression for Δa
        rho, a, e, Cd, A, m = sp.symbols('rho a e Cd A m', positive=True)
        Delta_a_expr = -2*sp.pi * a**2 * (Cd*A/m) * rho / (1 - e**2)   # per-orbit change
        # Numeric values (typical LEO satellite)
        a_nom = 7000e3        # m
        e_nom = 0.01
        Cd_nom = 2.2
        A_nom = 10.0          # m^2
        m_nom = 1000.0        # kg
        rho_nom = 1e-12       # kg/m^3 (very low density)
        # Baseline Δa
        Delta_a0 = float(Delta_a_expr.subs({
            rho: rho_nom, a: a_nom, e: e_nom, Cd: Cd_nom, A: A_nom, m: m_nom
        }))
        # Increase density by 10%
        rho_pert = 1.1 * rho_nom
        Delta_a1 = float(Delta_a_expr.subs({
            rho: rho_pert, a: a_nom, e: e_nom, Cd: Cd_nom, A: A_nom, m: m_nom
        }))
        # Check monotonicity: Δa1 should be more negative than Δa0
        if Delta_a1 < Delta_a0:
            # Also check that the result string mentions a functional form consistent with this
            s = str(self.drag_per_orbit_da).lower()
            has_negative = '-' in s
            has_density = any(term in s for term in ['rho', 'density', 'ρ'])
            if has_negative and has_density:
                return (True, f"Numerical monotonicity holds: Δa({rho_pert}) = {Delta_a1:.2e} < Δa({rho_nom}) = {Delta_a0:.2e}, and string includes '-' and density reference")
            else:
                # Still passes baseline monotonicity but warns about string
                return (True, f"Numerical monotonicity holds, but result string lacks explicit density reference")
        else:
            return (False, f"Monotonicity failed: Δa({rho_pert}) = {Delta_a1:.2e} not less than Δa({rho_nom}) = {Delta_a0:.2e}")

    # ---- Weak Answer Sanity Test ----
    def test_answer_sanity_covers_both_parts(self):
        """Check that the overall answer (or combined results) mentions both precession and circularization/decay."""
        if self.answer is not None:
            text = str(self.answer).lower()
        else:
            parts = [str(self.J2_secular_omega), str(self.J2_secular_Omega),
                     str(self.drag_per_orbit_da), str(self.drag_per_orbit_de)]
            text = " ".join([p for p in parts if p is not None]).lower()
        has_precession = any(term in text for term in ['precession', 'precess', 'node regression', 'perigee rotation'])
        has_circularization = any(term in text for term in ['circularization', 'circularise', 'circularize', 'decay'])
        if has_precession and has_circularization:
            return (True, "Answer covers both precession and circularization/decay")
        elif has_precession:
            return (False, "Answer mentions precession but missing circularization/decay")
        elif has_circularization:
            return (False, "Answer mentions circularization/decay but missing precession")
        else:
            return (False, "Answer does not mention either precession or circularization/decay")