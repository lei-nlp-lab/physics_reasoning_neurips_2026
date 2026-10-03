import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely – use None for missing values
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.m = getattr(solution_instance, 'm', givens.get('m', None))
        self.c_w = getattr(solution_instance, 'c_w', givens.get('c_w', None))
        self.c_i = getattr(solution_instance, 'c_i', givens.get('c_i', None))
        self.L_f = getattr(solution_instance, 'L_f', givens.get('L_f', None))
        # Temperatures (Kelvin assumed)
        self.T_initial = getattr(solution_instance, 'T_initial', givens.get('T_initial', None))
        self.T_fusion = getattr(solution_instance, 'T_fusion', givens.get('T_fusion', None))
        self.T_final = getattr(solution_instance, 'T_final', givens.get('T_final', None))
        self.T_res = getattr(solution_instance, 'T_res', givens.get('T_res', None))

        # Results dictionary
        self.results = getattr(solution_instance, 'results', {})
        # Extract the key result
        self.delta_S_univ = self.results.get('delta_S_univ', None)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_delta_S_univ_exists(self):
        """Check that delta_S_univ key is present in results."""
        if 'delta_S_univ' not in self.results:
            return (False, "delta_S_univ not found in results")
        return (True, "delta_S_univ key exists")

    def test_interface_delta_S_univ_type(self):
        """Check that delta_S_univ is numeric."""
        if self.delta_S_univ is None:
            return (False, "delta_S_univ is None")
        if not isinstance(self.delta_S_univ, (int, float)):
            return (False, f"delta_S_univ has type {type(self.delta_S_univ).__name__}, expected numeric")
        return (True, "delta_S_univ is numeric")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # ------------------------------------------------------------
    def test_invariant_entropy_of_universe_increases(self):
        """Second law: entropy of universe increases for an irreversible process.
        This cooling/freezing process is irreversible -> ΔS_univ > 0."""
        if self.delta_S_univ is None:
            return (False, "SKIP: delta_S_univ not available")
        # Allow small negative due to numerical noise
        if self.delta_S_univ <= -1e-9:
            return (False, f"delta_S_univ = {self.delta_S_univ} is negative, violates second law")
        return (True, f"delta_S_univ = {self.delta_S_univ} > 0 (within tolerance)")

    def test_invariant_temperatures_positive(self):
        """All absolute temperatures must be > 0 K."""
        temps = [self.T_initial, self.T_fusion, self.T_final, self.T_res]
        names = ["T_initial", "T_fusion", "T_final", "T_res"]
        for t, n in zip(temps, names):
            if t is None:
                return (False, f"SKIP: {n} is None")
            if t <= 0:
                return (False, f"{n} = {t} K is not positive")
        return (True, "All temperatures are positive")

    # ------------------------------------------------------------
    # Modeling / trace tests (consistency with physics equations)
    # ------------------------------------------------------------
    def test_modeling_additivity_delta_S(self):
        """Check additivity: ΔS_univ = ΔS_sys + ΔS_res.
        We compute ΔS_sys and ΔS_res from the givens and compare with reported ΔS_univ."""
        # Need all givens and delta_S_univ
        required = [self.m, self.c_w, self.c_i, self.L_f,
                    self.T_initial, self.T_fusion, self.T_final, self.T_res,
                    self.delta_S_univ]
        if any(v is None for v in required):
            return (False, "SKIP: missing required givens or delta_S_univ")

        # Compute ΔS_sys via reversible path
        try:
            delta_S_water = self.m * self.c_w * math.log(self.T_fusion / self.T_initial)
            delta_S_freeze = -self.m * self.L_f / self.T_fusion
            delta_S_ice = self.m * self.c_i * math.log(self.T_final / self.T_fusion)
            delta_S_sys = delta_S_water + delta_S_freeze + delta_S_ice

            # Compute ΔH_sys (in J)
            delta_H_sys = self.m * (self.c_w * (self.T_fusion - self.T_initial)
                                    - self.L_f
                                    + self.c_i * (self.T_final - self.T_fusion))
            # ΔS_res = -ΔH_sys / T_res
            delta_S_res = -delta_H_sys / self.T_res

            # Total from computation
            delta_S_univ_computed = delta_S_sys + delta_S_res
        except Exception as e:
            return (False, f"Computation failed: {e}")

        # Compare with reported value
        if not math.isclose(delta_S_univ_computed, self.delta_S_univ, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ΔS_univ computed = {delta_S_univ_computed}, reported = {self.delta_S_univ}, differ >2%")
        return (True, f"Additivity holds: ΔS_sys+ΔS_res = {delta_S_univ_computed} ≈ reported {self.delta_S_univ}")

    def test_modeling_system_entropy_decrease(self):
        """System loses entropy during cooling and freezing -> ΔS_sys < 0."""
        required = [self.m, self.c_w, self.c_i, self.L_f,
                    self.T_initial, self.T_fusion, self.T_final, self.T_res]
        if any(v is None for v in required):
            return (False, "SKIP: missing givens")

        try:
            delta_S_water = self.m * self.c_w * math.log(self.T_fusion / self.T_initial)
            delta_S_freeze = -self.m * self.L_f / self.T_fusion
            delta_S_ice = self.m * self.c_i * math.log(self.T_final / self.T_fusion)
            delta_S_sys = delta_S_water + delta_S_freeze + delta_S_ice
        except Exception as e:
            return (False, f"Computation failed: {e}")

        if delta_S_sys >= 0:
            return (False, f"ΔS_sys computed = {delta_S_sys} is not negative (system loses entropy)")
        return (True, f"ΔS_sys = {delta_S_sys} < 0")

    def test_modeling_reservoir_entropy_increase(self):
        """Reservoir absorbs heat -> its entropy increases: ΔS_res > 0."""
        required = [self.m, self.c_w, self.c_i, self.L_f,
                    self.T_initial, self.T_fusion, self.T_final, self.T_res]
        if any(v is None for v in required):
            return (False, "SKIP: missing givens")

        try:
            delta_H_sys = self.m * (self.c_w * (self.T_fusion - self.T_initial)
                                    - self.L_f
                                    + self.c_i * (self.T_final - self.T_fusion))
            delta_S_res = -delta_H_sys / self.T_res
        except Exception as e:
            return (False, f"Computation failed: {e}")

        if delta_S_res <= 0:
            return (False, f"ΔS_res computed = {delta_S_res} is not positive (reservoir gains entropy)")
        return (True, f"ΔS_res = {delta_S_res} > 0")

    # ------------------------------------------------------------
    # Monotonicity test (perturb initial temperature)
    # ------------------------------------------------------------
    def test_monotonicity_higher_initial_T_increases_delta_S_univ(self):
        """If initial water temperature is increased (e.g., by 1%),
        the magnitude of ΔS_sys becomes more negative (larger drop),
        but ΔS_res also changes; overall ΔS_univ should increase
        (more irreversibility)."""
        required = [self.m, self.c_w, self.c_i, self.L_f,
                    self.T_initial, self.T_fusion, self.T_final, self.T_res]
        if any(v is None for v in required):
            return (False, "SKIP: missing givens")

        # Baseline computation
        try:
            delta_S_water = self.m * self.c_w * math.log(self.T_fusion / self.T_initial)
            delta_S_freeze = -self.m * self.L_f / self.T_fusion
            delta_S_ice = self.m * self.c_i * math.log(self.T_final / self.T_fusion)
            delta_S_sys_base = delta_S_water + delta_S_freeze + delta_S_ice
            delta_H_sys_base = self.m * (self.c_w * (self.T_fusion - self.T_initial)
                                         - self.L_f
                                         + self.c_i * (self.T_final - self.T_fusion))
            delta_S_res_base = -delta_H_sys_base / self.T_res
            delta_S_univ_base = delta_S_sys_base + delta_S_res_base
        except Exception as e:
            return (False, f"Baseline computation failed: {e}")

        # Perturb T_initial upward by 1%
        T_initial_new = self.T_initial * 1.01
        try:
            delta_S_water_new = self.m * self.c_w * math.log(self.T_fusion / T_initial_new)
            # same freeze and ice terms
            delta_S_sys_new = delta_S_water_new + delta_S_freeze + delta_S_ice
            delta_H_sys_new = self.m * (self.c_w * (self.T_fusion - T_initial_new)
                                        - self.L_f
                                        + self.c_i * (self.T_final - self.T_fusion))
            delta_S_res_new = -delta_H_sys_new / self.T_res
            delta_S_univ_new = delta_S_sys_new + delta_S_res_new
        except Exception as e:
            return (False, f"Perturbed computation failed: {e}")

        # Expect that ΔS_univ increases (becomes more positive) with higher initial T
        if delta_S_univ_new <= delta_S_univ_base:
            return (False, f"ΔS_univ decreased from {delta_S_univ_base} to {delta_S_univ_new} after raising T_initial")
        return (True, f"ΔS_univ increased from {delta_S_univ_base} to {delta_S_univ_new} as expected")

    # ------------------------------------------------------------
    # Weak answer sanity (final answer magnitude)
    # ------------------------------------------------------------
    def test_answer_sanity_magnitude_range(self):
        """Weak sanity: ΔS_univ should be roughly 700-800 J/K (order 1e3)."""
        if self.delta_S_univ is None:
            return (False, "SKIP: delta_S_univ not available")
        # Broad range
        lower = 500.0
        upper = 1000.0
        if self.delta_S_univ < lower or self.delta_S_univ > upper:
            return (False, f"ΔS_univ = {self.delta_S_univ} outside expected range [{lower}, {upper}]")
        return (True, f"ΔS_univ = {self.delta_S_univ} within plausible range")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {e}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)