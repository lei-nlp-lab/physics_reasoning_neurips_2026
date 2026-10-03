import math
import traceback

class PhysicsChecker:
    """
    Conceptual problem checker for classical/quantum atomic emission.
    Includes unit tests for interface, invariants, modeling/trace,
    and monotonicity (by perturbing a numerical input if available).
    """

    def __init__(self, solution_instance):
        # Extract results safely; use None for ALL missing values.
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Core conceptual results
        self.classical_stability_possible = results.get('classical_stability_possible', None)
        self.qed_explains_spontaneous = results.get('qed_explains_spontaneous', None)
        self.emission_mechanism = results.get('emission_mechanism', None)

        # Optional numerical inputs for monotonicity test
        self.classical_orbit_radius = results.get('classical_orbit_radius', None)
        self.classical_electron_speed = results.get('classical_electron_speed', None)
        self.classical_electron_charge = results.get('classical_electron_charge', None)
        self.light_speed = results.get('light_speed', None)

        # Explanation steps (optional)
        self.explanation_steps = None
        steps = getattr(solution_instance, 'explanation_steps', None)
        if steps is not None:
            self.explanation_steps = str(steps)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_classical_key_exists(self):
        if self.classical_stability_possible is None:
            return (False, "Missing result 'classical_stability_possible' (expected bool)")
        if not isinstance(self.classical_stability_possible, bool):
            return (False, "classical_stability_possible must be a bool")
        return (True, "classical_stability_possible present and bool")

    def test_interface_quantum_key_exists(self):
        if self.qed_explains_spontaneous is None:
            return (False, "Missing result 'qed_explains_spontaneous' (expected bool)")
        if not isinstance(self.qed_explains_spontaneous, bool):
            return (False, "qed_explains_spontaneous must be a bool")
        return (True, "qed_explains_spontaneous present and bool")

    def test_interface_mechanism_key_exists(self):
        if self.emission_mechanism is None:
            return (False, "Missing result 'emission_mechanism' (expected str)")
        if not isinstance(self.emission_mechanism, str):
            return (False, "emission_mechanism must be a string")
        return (True, "emission_mechanism present and str")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physics sanity)
    # ------------------------------------------------------------------
    def test_invariant_classical_radiation_inevitable(self):
        if self.classical_stability_possible is None:
            return (False, "SKIP: classical_stability_possible not available")
        if self.classical_stability_possible is not False:
            return (False, "classical_stability_possible must be False (Larmor formula forces energy loss)")
        return (True, "classical_stability_possible is False, consistent with classical physics")

    def test_invariant_quantum_excited_atoms_decay(self):
        if self.qed_explains_spontaneous is None:
            return (False, "SKIP: qed_explains_spontaneous not available")
        if self.qed_explains_spontaneous is not True:
            return (False, "qed_explains_spontaneous must be True (QED predicts spontaneous emission)")
        return (True, "qed_explains_spontaneous is True, consistent with QED")

    def test_invariant_emission_mechanism_requires_creation_operator(self):
        if self.emission_mechanism is None:
            return (False, "SKIP: emission_mechanism not available")
        mechanism_lower = self.emission_mechanism.lower()
        keywords = ['creation', 'photon creation', 'a^†', 'creation operator']
        if not any(kw in mechanism_lower for kw in keywords):
            return (False, f"Emission mechanism '{self.emission_mechanism}' must mention photon creation")
        return (True, "Emission mechanism correctly involves single‑photon creation operator")

    # ------------------------------------------------------------------
    # Modeling / trace consistency (using explanation_steps)
    # ------------------------------------------------------------------
    def test_modeling_reasoning_mentions_instability(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        steps_lower = self.explanation_steps.lower()
        concepts = ['larmor', 'acceleration', 'radiat', 'spiral', 'energy loss']
        found = [c for c in concepts if c in steps_lower]
        if len(found) < 2:
            return (False, f"Explanation missing key classical concepts; found only {found}")
        return (True, f"Explanation references classical instability: {found}")

    def test_modeling_reasoning_mentions_qed_mechanism(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        steps_lower = self.explanation_steps.lower()
        concepts = ['qed', 'quantum electrodynam', 'vacuum', 'creation operator', 'a^†']
        found = [c for c in concepts if c in steps_lower]
        if len(found) < 1:
            return (False, f"Explanation missing QED/creation mechanism; found only {found}")
        return (True, f"Explanation references QED/creation: {found}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_classical_power_decreases_with_radius(self):
        """
        Classical Larmor formula: P ∝ a^2 ∝ v^4 / r^2 (since a = v^2/r).
        If orbit radius increases (with same speed), radiated power decreases.
        This test perturbs the radius by +10% and verifies power decreases.
        If any required numerical input is missing, the test is skipped.
        """
        # Check that all required numerical quantities are provided
        if any(x is None for x in [self.classical_orbit_radius,
                                   self.classical_electron_speed,
                                   self.classical_electron_charge,
                                   self.light_speed]):
            return (False, "SKIP: missing one or more of classical_orbit_radius, "
                           "classical_electron_speed, classical_electron_charge, light_speed")

        charge = self.classical_electron_charge
        c = self.light_speed
        # Baseline values
        r0 = self.classical_orbit_radius
        v0 = self.classical_electron_speed
        a0 = v0**2 / r0
        P0 = (2./3.) * (charge**2 * a0**2) / c**3

        # Perturb radius: increase by 10%
        r1 = r0 * 1.10
        a1 = v0**2 / r1
        P1 = (2./3.) * (charge**2 * a1**2) / c**3

        # Check monotonicity: power should strictly decrease
        if P1 >= P0:
            return (False, f"Radiated power should decrease when radius increases; "
                           f"P0={P0:.4e}, P1={P1:.4e}")
        # Also verify the ratio P1/P0 matches (r0/r1)^2
        expected_ratio = (r0 / r1)**2
        actual_ratio = P1 / P0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-9):
            return (False, f"Power ratio {actual_ratio:.6f} does not match expected {expected_ratio:.6f}")
        return (True, f"Radiated power decreases (P0={P0:.4e} -> P1={P1:.4e}) with radius increase")

    # ------------------------------------------------------------------
    # Weak answer‑sanity (at most one, no hardcoded targets)
    # ------------------------------------------------------------------
    def test_answer_sanity_classical_boolean_false(self):
        if self.classical_stability_possible is None:
            return (False, "SKIP: classical_stability_possible not provided")
        if self.classical_stability_possible is not False:
            return (False, "classical_stability_possible must be False (classical instability)")
        return (True, "classical_stability_possible is False")

    # ------------------------------------------------------------------
    # Runner
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