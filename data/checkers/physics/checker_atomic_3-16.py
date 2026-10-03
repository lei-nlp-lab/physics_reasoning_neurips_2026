import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.delta_Lmu = results.get('delta_Lmu', None)
        self.delta_Le = results.get('delta_Le', None)
        self.delta_Q = results.get('delta_Q', None)
        self.delta_S = results.get('delta_S', None)
        self.delta_B = results.get('delta_B', None)
        self.mass_sum = results.get('mass_sum', None)
        self.m_p = results.get('m_p', None)
        self.conserved = results.get('conserved', None)

        # Particle quantum numbers (ground truth)
        self.particles = {
            'mu+':   {'Q': +1, 'B': 0, 'Le': 0, 'Lmu': -1, 'S': 0},
            'mu-':   {'Q': -1, 'B': 0, 'Le': 0, 'Lmu': +1, 'S': 0},
            'e+':    {'Q': +1, 'B': 0, 'Le': -1, 'Lmu': 0, 'S': 0},
            'e-':    {'Q': -1, 'B': 0, 'Le': +1, 'Lmu': 0, 'S': 0},
            'gamma': {'Q': 0, 'B': 0, 'Le': 0, 'Lmu': 0, 'S': 0},
            'nu_e':  {'Q': 0, 'B': 0, 'Le': +1, 'Lmu': 0, 'S': 0},
            'nu_mu': {'Q': 0, 'B': 0, 'Le': 0, 'Lmu': +1, 'S': 0},
            'p':     {'Q': +1, 'B': +1, 'Le': 0, 'Lmu': 0, 'S': 0},
            'n':     {'Q': 0, 'B': +1, 'Le': 0, 'Lmu': 0, 'S': 0},
            'Sigma+':{'Q': +1, 'B': +1, 'Le': 0, 'Lmu': 0, 'S': -1},
            'K-':    {'Q': -1, 'B': 0, 'Le': 0, 'Lmu': 0, 'S': -1},
            'pi+':   {'Q': +1, 'B': 0, 'Le': 0, 'Lmu': 0, 'S': 0},
            'nu_bar_e':{'Q': 0, 'B': 0, 'Le': -1, 'Lmu': 0, 'S': 0},
        }

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_delta_Lmu(self):
        if self.delta_Lmu is None:
            return (False, "SKIP: delta_Lmu not provided")
        if not isinstance(self.delta_Lmu, (int, float)):
            return (False, f"delta_Lmu is not numeric: {self.delta_Lmu}")
        return (True, "delta_Lmu is present and numeric")

    def test_interface_delta_Le(self):
        if self.delta_Le is None:
            return (False, "SKIP: delta_Le not provided")
        if not isinstance(self.delta_Le, (int, float)):
            return (False, f"delta_Le is not numeric: {self.delta_Le}")
        return (True, "delta_Le is present and numeric")

    def test_interface_delta_Q(self):
        if self.delta_Q is None:
            return (False, "SKIP: delta_Q not provided")
        if not isinstance(self.delta_Q, (int, float)):
            return (False, f"delta_Q is not numeric: {self.delta_Q}")
        return (True, "delta_Q is present and numeric")

    def test_interface_delta_S(self):
        if self.delta_S is None:
            return (False, "SKIP: delta_S not provided")
        if not isinstance(self.delta_S, (int, float)):
            return (False, f"delta_S is not numeric: {self.delta_S}")
        return (True, "delta_S is present and numeric")

    def test_interface_delta_B(self):
        if self.delta_B is None:
            return (False, "SKIP: delta_B not provided")
        if not isinstance(self.delta_B, (int, float)):
            return (False, f"delta_B is not numeric: {self.delta_B}")
        return (True, "delta_B is present and numeric")

    def test_interface_mass_sum(self):
        if self.mass_sum is None:
            return (False, "SKIP: mass_sum not provided")
        if not isinstance(self.mass_sum, (int, float)):
            return (False, f"mass_sum is not numeric: {self.mass_sum}")
        return (True, "mass_sum is present and numeric")

    def test_interface_m_p(self):
        if self.m_p is None:
            return (False, "SKIP: m_p not provided")
        if not isinstance(self.m_p, (int, float)):
            return (False, f"m_p is not numeric: {self.m_p}")
        return (True, "m_p is present and numeric")

    def test_interface_conserved(self):
        if self.conserved is None:
            return (False, "SKIP: conserved not provided")
        if not isinstance(self.conserved, bool):
            return (False, f"conserved is not boolean: {self.conserved}")
        return (True, "conserved is present and boolean")

    # ------------------------------------------------------------------
    # Invariant tests
    # ------------------------------------------------------------------
    def test_invariant_delta_integer(self):
        deltas = [
            ('delta_Lmu', self.delta_Lmu),
            ('delta_Le', self.delta_Le),
            ('delta_Q', self.delta_Q),
            ('delta_S', self.delta_S),
            ('delta_B', self.delta_B),
        ]
        for name, val in deltas:
            if val is None:
                continue
            if not math.isclose(val, round(val), rel_tol=0, abs_tol=1e-9):
                return (False, f"{name} = {val} is not close to integer")
        return (True, "All present deltas are integers (within tolerance)")

    def test_invariant_conservation_consistency(self):
        if self.conserved is None:
            return (False, "SKIP: conserved not provided")
        deltas = [self.delta_Lmu, self.delta_Le, self.delta_Q, self.delta_S, self.delta_B]
        deltas = [d for d in deltas if d is not None]
        if not deltas:
            return (False, "SKIP: no delta values provided")
        if self.conserved:
            non_zero = [d for d in deltas if not math.isclose(d, 0, abs_tol=1e-9)]
            if non_zero:
                return (False, f"conserved=True but deltas non-zero: {non_zero}")
            return (True, "conserved=True and all deltas are zero")
        else:
            zero = [d for d in deltas if math.isclose(d, 0, abs_tol=1e-9)]
            if len(zero) == len(deltas):
                return (False, "conserved=False but all deltas are zero")
            return (True, "conserved=False and at least one delta non-zero")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------
    def _compute_delta(self, reaction):
        init_Q = 0; init_B = 0; init_Le = 0; init_Lmu = 0; init_S = 0
        final_Q = 0; final_B = 0; final_Le = 0; final_Lmu = 0; final_S = 0
        for name, is_init in reaction:
            p = self.particles.get(name)
            if p is None:
                raise ValueError(f"Unknown particle: {name}")
            if is_init:
                init_Q += p['Q']
                init_B += p['B']
                init_Le += p['Le']
                init_Lmu += p['Lmu']
                init_S += p['S']
            else:
                final_Q += p['Q']
                final_B += p['B']
                final_Le += p['Le']
                final_Lmu += p['Lmu']
                final_S += p['S']
        return {
            'Q': init_Q - final_Q,
            'B': init_B - final_B,
            'Le': init_Le - final_Le,
            'Lmu': init_Lmu - final_Lmu,
            'S': init_S - final_S,
        }

    def test_modeling_reaction1(self):
        if self.delta_Lmu is None or self.delta_Le is None:
            return (False, "SKIP: delta_Lmu/delta_Le not present")
        try:
            expected = self._compute_delta([
                ('mu+', True),
                ('e+', False),
                ('gamma', False)
            ])
        except ValueError as e:
            return (False, f"Modeling error: {e}")
        for name, exp_val in [('delta_Lmu', expected['Lmu']), ('delta_Le', expected['Le'])]:
            got = self.delta_Lmu if name == 'delta_Lmu' else self.delta_Le
            if not math.isclose(got, exp_val, abs_tol=1e-9):
                return (False, f"{name} expected {exp_val}, got {got}")
        return (True, "Reaction 1 (mu+ -> e+ + gamma) deltas match")

    def test_modeling_reaction2(self):
        if self.delta_Q is None:
            return (False, "SKIP: delta_Q not present")
        try:
            expected = self._compute_delta([
                ('e-', True),
                ('nu_e', False),
                ('gamma', False)
            ])
        except ValueError as e:
            return (False, f"Modeling error: {e}")
        if not math.isclose(self.delta_Q, expected['Q'], abs_tol=1e-9):
            return (False, f"delta_Q expected {expected['Q']}, got {self.delta_Q}")
        return (True, "Reaction 2 (e- -> nu_e + gamma) delta_Q matches")

    def test_modeling_reaction3(self):
        if self.delta_Q is None or self.delta_S is None:
            return (False, "SKIP: delta_Q/delta_S not present")
        try:
            expected = self._compute_delta([
                ('p', True), ('p', True),
                ('p', False), ('Sigma+', False), ('K-', False)
            ])
        except ValueError as e:
            return (False, f"Modeling error: {e}")
        for name, exp_val in [('delta_Q', expected['Q']), ('delta_S', expected['S'])]:
            got = self.delta_Q if name == 'delta_Q' else self.delta_S
            if not math.isclose(got, exp_val, abs_tol=1e-9):
                return (False, f"{name} expected {exp_val}, got {got}")
        return (True, "Reaction 3 (p+p -> p+Sigma++K-) deltas match")

    def test_modeling_reaction4(self):
        if self.delta_B is None:
            return (False, "SKIP: delta_B not present")
        try:
            expected = self._compute_delta([
                ('p', True),
                ('e+', False), ('nu_e', False)
            ])
        except ValueError as e:
            return (False, f"Modeling error: {e}")
        if not math.isclose(self.delta_B, expected['B'], abs_tol=1e-9):
            return (False, f"delta_B expected {expected['B']}, got {self.delta_B}")
        return (True, "Reaction 4 (p -> e+ + nu_e) delta_B matches")

    def test_modeling_reaction5(self):
        if self.mass_sum is None or self.m_p is None:
            return (False, "SKIP: mass_sum/m_p not present")
        if not math.isclose(self.m_p, 938.272, rel_tol=1e-4):
            return (False, f"m_p = {self.m_p} not close to 938.272")
        expected_sum = 0.511 + 939.565
        if not math.isclose(self.mass_sum, expected_sum, rel_tol=1e-4):
            return (False, f"mass_sum = {self.mass_sum} not close to {expected_sum}")
        if self.m_p >= self.mass_sum - 1e-9:
            return (False, "Energy conservation would be satisfied, but expected violation")
        return (True, "Reaction 5 (p -> e+ + n + nu_e) mass values and violation consistent")

    def test_modeling_reaction6(self):
        if self.conserved is None:
            return (False, "SKIP: conserved not present")
        if self.conserved is not True:
            return (False, "Expected conserved=True for neutron decay")
        return (True, "Reaction 6 (n decay) marked as conserved")

    def test_modeling_reaction7(self):
        if self.delta_Lmu is None or self.delta_Le is None:
            return (False, "SKIP: delta_Lmu/delta_Le not present")
        try:
            expected = self._compute_delta([
                ('pi+', True),
                ('mu+', False), ('nu_mu', False)
            ])
        except ValueError as e:
            return (False, f"Modeling error: {e}")
        if not math.isclose(self.delta_Lmu, expected['Lmu'], abs_tol=1e-9):
            return (False, f"delta_Lmu expected {expected['Lmu']}, got {self.delta_Lmu}")
        if not math.isclose(self.delta_Le, expected['Le'], abs_tol=1e-9):
            return (False, f"delta_Le expected {expected['Le']}, got {self.delta_Le}")
        return (True, "Reaction 7 (pi+ -> mu+ + nu_mu) lepton deltas zero")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation and comparison)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_violation(self):
        # Perturb m_p upward and compare violation flag
        if self.m_p is None or self.mass_sum is None:
            return (False, "SKIP: m_p or mass_sum not provided")
        baseline_violation = self.m_p < self.mass_sum
        perturbed_m_p = self.m_p * 1.01
        perturbed_violation = perturbed_m_p < self.mass_sum
        if not baseline_violation and perturbed_violation:
            return (False, "Monotonicity failure: increasing m_p turned non-violation into violation")
        return (True, "Monotonicity check passed: increasing m_p does not create energy violation")

    def test_monotonicity_delta_sign(self):
        """
        For a present and non-zero delta, perturb the initial state by adding one extra particle
        that contributes positively to that quantum number. Verify that the delta changes
        in the expected direction (magnitude increases or sign preserved).
        Uses the computed delta from modeling as baseline.
        """
        # Try to identify the reaction and corresponding delta based on which keys are present
        # We'll handle only cases where we have clear mapping.
        # Case 1: delta_Lmu and delta_Le present and non-zero -> reaction1 (mu+ -> e+ gamma)
        if (self.delta_Lmu is not None and not math.isclose(self.delta_Lmu, 0, abs_tol=1e-9) and
            self.delta_Le is not None and not math.isclose(self.delta_Le, 0, abs_tol=1e-9)):
            # Baseline computation (same as modeling test)
            try:
                base_reaction = [('mu+', True), ('e+', False), ('gamma', False)]
                base_delta_Lmu = self._compute_delta(base_reaction)['Lmu']
            except ValueError as e:
                return (False, f"Modeling error: {e}")
            # Perturb by adding an extra initial mu+
            perturbed_reaction = [('mu+', True), ('mu+', True), ('e+', False), ('gamma', False)]
            perturbed_delta_Lmu = self._compute_delta(perturbed_reaction)['Lmu']
            # Expect perturbed = base + 1 (since mu+ contributes -1 to Lmu, initial Lmu decreases by 1, so delta increases by 1)
            expected_perturbed = base_delta_Lmu + 1
            if not math.isclose(perturbed_delta_Lmu, expected_perturbed, abs_tol=1e-9):
                return (False, f"For reaction1: perturbed delta_Lmu = {perturbed_delta_Lmu}, expected {expected_perturbed}")
            return (True, "Monotonicity for reaction1: adding initial mu+ increases delta_Lmu by 1")

        # Case 2: delta_Q present and non-zero, delta_S absent or zero -> reaction2 (e- -> nu_e gamma)
        if (self.delta_Q is not None and not math.isclose(self.delta_Q, 0, abs_tol=1e-9) and
            (self.delta_S is None or math.isclose(self.delta_S, 0, abs_tol=1e-9))):
            try:
                base_reaction = [('e-', True), ('nu_e', False), ('gamma', False)]
                base_delta_Q = self._compute_delta(base_reaction)['Q']
            except ValueError as e:
                return (False, f"Modeling error: {e}")
            # Perturb by adding an extra initial e- (which has Q=-1, so initial Q decreases by 1, delta_Q increases by 1)
            perturbed_reaction = [('e-', True), ('e-', True), ('nu_e', False), ('gamma', False)]
            perturbed_delta_Q = self._compute_delta(perturbed_reaction)['Q']
            expected_perturbed = base_delta_Q + 1  # base_delta_Q is -1, so becomes 0
            if not math.isclose(perturbed_delta_Q, expected_perturbed, abs_tol=1e-9):
                return (False, f"For reaction2: perturbed delta_Q = {perturbed_delta_Q}, expected {expected_perturbed}")
            return (True, "Monotonicity for reaction2: adding initial e- increases delta_Q by 1")

        # Case 3: delta_Q and delta_S both present and non-zero -> reaction3 (p+p -> p+Sigma++K-)
        if (self.delta_Q is not None and not math.isclose(self.delta_Q, 0, abs_tol=1e-9) and
            self.delta_S is not None and not math.isclose(self.delta_S, 0, abs_tol=1e-9)):
            try:
                base_reaction = [('p', True), ('p', True), ('p', False), ('Sigma+', False), ('K-', False)]
                base_delta_Q = self._compute_delta(base_reaction)['Q']
                base_delta_S = self._compute_delta(base_reaction)['S']
            except ValueError as e:
                return (False, f"Modeling error: {e}")
            # Perturb by adding an extra initial p (Q=+1, S=0)
            perturbed_reaction = [('p', True), ('p', True), ('p', True), ('p', False), ('Sigma+', False), ('K-', False)]
            perturbed_delta_Q = self._compute_delta(perturbed_reaction)['Q']
            perturbed_delta_S = self._compute_delta(perturbed_reaction)['S']
            expected_perturbed_Q = base_delta_Q + 1  # adding initial p increases initial Q by 1, so delta increases by 1
            expected_perturbed_S = base_delta_S      # S unchanged
            if not math.isclose(perturbed_delta_Q, expected_perturbed_Q, abs_tol=1e-9):
                return (False, f"For reaction3: perturbed delta_Q = {perturbed_delta_Q}, expected {expected_perturbed_Q}")
            if not math.isclose(perturbed_delta_S, expected_perturbed_S, abs_tol=1e-9):
                return (False, f"For reaction3: perturbed delta_S = {perturbed_delta_S}, expected {expected_perturbed_S}")
            return (True, "Monotonicity for reaction3: adding initial p increases delta_Q by 1, delta_S unchanged")

        # Case 4: delta_B present and non-zero -> reaction4 (p -> e+ nu_e)
        if self.delta_B is not None and not math.isclose(self.delta_B, 0, abs_tol=1e-9):
            try:
                base_reaction = [('p', True), ('e+', False), ('nu_e', False)]
                base_delta_B = self._compute_delta(base_reaction)['B']
            except ValueError as e:
                return (False, f"Modeling error: {e}")
            # Perturb by adding an extra initial p (B=+1)
            perturbed_reaction = [('p', True), ('p', True), ('e+', False), ('nu_e', False)]
            perturbed_delta_B = self._compute_delta(perturbed_reaction)['B']
            expected_perturbed = base_delta_B + 1  # base is 1, becomes 2
            if not math.isclose(perturbed_delta_B, expected_perturbed, abs_tol=1e-9):
                return (False, f"For reaction4: perturbed delta_B = {perturbed_delta_B}, expected {expected_perturbed}")
            return (True, "Monotonicity for reaction4: adding initial p increases delta_B by 1")

        # If no matching case, skip
        return (False, "SKIP: No suitable non-zero delta found for monotonicity perturbation test")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_general(self):
        deltas = [self.delta_Lmu, self.delta_Le, self.delta_Q, self.delta_S, self.delta_B]
        for d in deltas:
            if d is None:
                continue
            if not math.isfinite(d):
                return (False, f"Delta is not finite: {d}")
            if abs(d) > 10:
                return (False, f"Delta magnitude too large: {d}")
        for val in [self.mass_sum, self.m_p]:
            if val is None:
                continue
            if not math.isfinite(val) or val <= 0:
                return (False, f"Mass value invalid: {val}")
        if self.conserved is not None and not isinstance(self.conserved, bool):
            return (False, f"conserved is not boolean: {self.conserved}")
        return (True, "Basic sanity checks passed")

    # ------------------------------------------------------------------
    # Main runner
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