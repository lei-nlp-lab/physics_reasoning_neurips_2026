import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a two-body elastic collision problem.
    Checks correctness of statements (a), (b), (c) about trajectories, helicity,
    and angular distribution in the center-of-mass frame.
    """

    def __init__(self, solution_instance):
        # Extract results safely; default to None if missing
        results = getattr(solution_instance, 'results', {})
        self.a = results.get('a', None)
        self.b = results.get('b', None)
        self.c = results.get('c', None)

        # Auxiliary information for reasoning checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None)
        self.key_constraints = getattr(solution_instance, 'key_constraints', None)

    # ----- Interface tests -----

    def test_interface_required_keys(self):
        """Check that all required result keys (a, b, c) are present."""
        missing = []
        if self.a is None:
            missing.append('a')
        if self.b is None:
            missing.append('b')
        if self.c is None:
            missing.append('c')
        if missing:
            return False, f"Missing required key(s): {', '.join(missing)}"
        return True, "All required keys present"

    def test_interface_results_are_bool(self):
        """Check that a, b, c are boolean values."""
        if self.a is None or self.b is None or self.c is None:
            return False, "Results not fully available; cannot check type"
        for key, val in [('a', self.a), ('b', self.b), ('c', self.c)]:
            if not isinstance(val, bool):
                return False, f"Result '{key}' is not a boolean (got {type(val).__name__})"
        return True, "All results are boolean"

    # ----- Invariant tests (theorem-based physics sanity) -----

    def test_invariant_momentum_conservation(self):
        """Verify that momentum conservation in CM frame is referenced in physics mapping or explanation."""
        text = ""
        if self.physics_mapping is not None:
            text += str(self.physics_mapping) + " "
        if self.explanation_steps is not None:
            text += str(self.explanation_steps) + " "
        if not text.strip():
            return False, "SKIP: No physics_mapping or explanation_steps available"
        keywords = [
            "momentum conservation",
            "p1_i = -p2_i",
            "p1_f = -p2_f",
            "total momentum zero",
            "cm frame total momentum zero",
        ]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return False, "Momentum conservation not mentioned in physics mapping or explanation"
        return True, "Momentum conservation referenced"

    def test_invariant_energy_conservation(self):
        """Verify that elastic collision (kinetic energy conservation) is referenced."""
        text = ""
        if self.physics_mapping is not None:
            text += str(self.physics_mapping) + " "
        if self.explanation_steps is not None:
            text += str(self.explanation_steps) + " "
        if not text.strip():
            return False, "SKIP: No physics_mapping or explanation_steps available"
        keywords = [
            "energy conservation",
            "kinetic energy conserved",
            "elastic collision",
            "|p1_i|^2/(2m1) + |p2_i|^2/(2m2) = |p1_f|^2/(2m1) + |p2_f|^2/(2m2)"
        ]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return False, "Energy conservation not mentioned in physics mapping or explanation"
        return True, "Energy conservation referenced"

    # ----- Modeling / trace checks (quality of reasoning) -----

    def test_modeling_plane_trajectories(self):
        """Check that the explanation includes the plane argument for statement (a)."""
        if self.explanation_steps is None:
            return False, "SKIP: No explanation_steps available"
        text = str(self.explanation_steps)
        keywords = [
            "plane",
            "trajectories lie in the same plane",
            "all trajectories in one plane",
            "initial and final momenta define a plane"
        ]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return False, "Explanation for (a) missing the plane argument"
        return True, "Plane reasoning found in explanation"

    def test_modeling_helicity_reasoning(self):
        """Check that the explanation for (b) correctly states that helicity can change."""
        if self.explanation_steps is None:
            return False, "SKIP: No explanation_steps available"
        text = str(self.explanation_steps)
        keywords = [
            "helicity can change",
            "spin dependent",
            "counterexample",
            "spin-flip",
            "no general conservation"
        ]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return False, "Explanation for (b) missing argument that helicity can change"
        return True, "Helicity change reasoning found"

    def test_modeling_angular_distribution(self):
        """Check that the explanation for (c) correctly states that the angular distribution is not generally symmetric."""
        if self.explanation_steps is None:
            return False, "SKIP: No explanation_steps available"
        text = str(self.explanation_steps)
        keywords = [
            "not spherically symmetric",
            "not isotropic",
            "angular distribution depends",
            "generally false",
            "dsigma/dOmega not constant"
        ]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return False, "Explanation for (c) missing argument that angular distribution is not symmetric"
        return True, "Angular distribution reasoning found"

    # ----- Monotonicity test (explicit perturbation and comparison) -----

    def test_monotonicity_scattering_angle_vs_impact_parameter(self):
        """
        For two-body elastic collisions with a central potential, the scattering
        angle in the CM frame is a monotonic decreasing function of the impact
        parameter b (for repulsive potentials). This test verifies that property
        using a hard-sphere model (radius d) as a concrete example.
        """
        # Hard‑sphere radius (any positive number)
        d = 1.0

        # Baseline impact parameter
        b_base = 0.3 * d
        # Perturbed (larger) impact parameter
        b_pert = 0.5 * d

        # Deflection angle for hard‑sphere: theta = 2 * acos(b/d) for b <= d
        def deflection_angle(b, d):
            if b >= d:
                return 0.0
            return 2.0 * math.acos(b / d)

        theta_base = deflection_angle(b_base, d)
        theta_pert = deflection_angle(b_pert, d)

        # Monotonicity: larger b → smaller theta
        if theta_base <= theta_pert:
            return (False,
                    f"Monotonicity violation: theta(b={b_base})={theta_base:.4f} "
                    f"should be > theta(b={b_pert})={theta_pert:.4f} "
                    f"(b2 > b1 but theta did not decrease)")
        return True, "Scattering angle decreases monotonically with impact parameter"

    # ----- Weak answer sanity (only one such method allowed) -----

    def test_answer_sanity_weak(self):
        """
        Weak sanity: verify the overall answer pattern.
        Only (a) should be True; (b) and (c) should be False.
        This is a simple boolean check, not a re-derivation.
        """
        if self.a is None or self.b is None or self.c is None:
            return False, "Missing results, cannot perform sanity check"
        if not all(isinstance(x, bool) for x in [self.a, self.b, self.c]):
            return False, "Results are not all boolean"
        true_count = sum([self.a, self.b, self.c])
        if true_count != 1:
            return False, f"Expected exactly 1 true statement, got {true_count}"
        if self.a is not True:
            return False, "Statement (a) should be True"
        if self.b is not False:
            return False, "Statement (b) should be False"
        if self.c is not False:
            return False, "Statement (c) should be False"
        return True, "Answer passes weak sanity: (a)=True, (b)=False, (c)=False"

    # ----- Test runner -----

    def run_all_tests(self):
        """
        Discover and run all test_ methods, collecting results.
        Returns (success, logs) where logs is a list of strings.
        """
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