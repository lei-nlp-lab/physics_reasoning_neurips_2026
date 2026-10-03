import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Required result keys (emit contract)
        self.L = self.results.get('L', None)
        self.H_w = self.results.get('H_w', None)
        self.G_w = self.results.get('G_w', None)
        self.G_s = self.results.get('G_s', None)
        self.dG = self.results.get('dG', None)

        # Given physical parameters (extract from solution_instance attributes)
        self.T = getattr(solution_instance, 'T', None) or getattr(solution_instance, 'temperature', None) or None
        self.S_w = getattr(solution_instance, 'S_w', None) or getattr(solution_instance, 'entropy_water', None) or None
        self.S_s = getattr(solution_instance, 'S_s', None) or getattr(solution_instance, 'entropy_steam', None) or None
        self.H_s = getattr(solution_instance, 'H_s', None) or getattr(solution_instance, 'enthalpy_steam', None) or None

        # Convert to numeric if possible
        for attr in ['T', 'S_w', 'S_s', 'H_s']:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except (ValueError, TypeError):
                    setattr(self, attr, None)

    # --------------------------------------------------------------------------
    # Interface checks: ensure required keys exist and are valid numbers
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self):
        missing = []
        for key in ['L', 'H_w', 'G_w', 'G_s', 'dG']:
            val = getattr(self, key, None)
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required result keys: {missing}")
        # Check numeric type
        for key in ['L', 'H_w', 'G_w', 'G_s', 'dG']:
            val = getattr(self, key)
            if not isinstance(val, (int, float)):
                return (False, f"Key '{key}' is not numeric (got {type(val).__name__})")
        return (True, "All required keys present and numeric")

    def test_interface_givens_available(self):
        missing = []
        for name in ['T', 'S_w', 'S_s', 'H_s']:
            if getattr(self, name, None) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing given parameter(s): {missing}")
        return (True, "All given parameters available")

    # --------------------------------------------------------------------------
    # Invariant checks: theorem-based physics relationships (use isclose)
    # --------------------------------------------------------------------------
    def test_invariant_gibbs_equality(self):
        """At phase equilibrium (reversible isothermal isobaric), ΔG = 0.
           Thus G_water ≈ G_steam."""
        if self.G_w is None or self.G_s is None:
            return (False, "SKIP: G_w or G_s not available")
        if not math.isclose(self.G_w, self.G_s, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"G_w ({self.G_w}) and G_s ({self.G_s}) differ; expected ΔG≈0")
        return (True, f"G_w = {self.G_w}, G_s = {self.G_s}, ΔG ≈ 0")

    def test_invariant_dG_zero(self):
        """dG should be exactly 0 for reversible isothermal isobaric process."""
        if self.dG is None:
            return (False, "SKIP: dG not available")
        if not math.isclose(self.dG, 0.0, abs_tol=1e-2):
            return (False, f"dG = {self.dG} but expected 0")
        return (True, f"dG = {self.dG} ≈ 0")

    def test_invariant_heat_of_vaporization(self):
        """L = T * ΔS = T*(S_s - S_w)  (Clausius–Clapeyron for phase equilibrium)"""
        if self.L is None or self.T is None or self.S_w is None or self.S_s is None:
            return (False, "SKIP: L, T, S_w, or S_s missing")
        delta_S = self.S_s - self.S_w
        expected_L = self.T * delta_S
        if not math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=0.5):
            return (False, f"L = {self.L}, but expected T*ΔS = {expected_L:.2f}")
        return (True, f"L = {self.L} matches T*ΔS = {expected_L:.2f}")

    def test_invariant_enthalpy_consistency(self):
        """H_s - H_w = L (energy balance for phase change)"""
        if self.H_w is None or self.H_s is None or self.L is None:
            return (False, "SKIP: H_w, H_s, or L missing")
        expected_diff = self.H_s - self.H_w
        if not math.isclose(expected_diff, self.L, rel_tol=1e-2, abs_tol=0.5):
            return (False, f"H_s - H_w = {expected_diff:.2f} but L = {self.L}")
        return (True, f"H_s - H_w = {expected_diff:.2f} ≈ L = {self.L}")

    # --------------------------------------------------------------------------
    # Modeling checks: verify that Gibbs free energy is computed correctly
    # --------------------------------------------------------------------------
    def test_modeling_gibbs_water_definition(self):
        """G_w = H_w - T * S_w"""
        if self.G_w is None or self.H_w is None or self.T is None or self.S_w is None:
            return (False, "SKIP: missing data for G_w computation")
        expected = self.H_w - self.T * self.S_w
        if not math.isclose(self.G_w, expected, rel_tol=1e-2, abs_tol=1.0):
            return (False, f"G_w = {self.G_w} but H_w - T*S_w = {expected:.2f}")
        return (True, f"G_w = {self.G_w} matches H_w - T*S_w = {expected:.2f}")

    def test_modeling_gibbs_steam_definition(self):
        """G_s = H_s - T * S_s"""
        if self.G_s is None or self.H_s is None or self.T is None or self.S_s is None:
            return (False, "SKIP: missing data for G_s computation")
        expected = self.H_s - self.T * self.S_s
        if not math.isclose(self.G_s, expected, rel_tol=1e-2, abs_tol=1.0):
            return (False, f"G_s = {self.G_s} but H_s - T*S_s = {expected:.2f}")
        return (True, f"G_s = {self.G_s} matches H_s - T*S_s = {expected:.2f}")

    # --------------------------------------------------------------------------
    # Monotonicity checks: perturb T and verify G decreases
    # --------------------------------------------------------------------------
    def test_monotonicity_gibbs_decreases_with_temperature(self):
        """For a given substance, dG/dT = -S < 0, so increasing T decreases G.
           We perturb T by +5% and check that recomputed G_s becomes lower."""
        if self.G_s is None or self.H_s is None or self.T is None or self.S_s is None:
            return (False, "SKIP: missing data for monotonicity check on G_s")
        # Baseline G_s
        baseline = self.G_s
        # Perturb temperature upward by 5%
        T_perturbed = self.T * 1.05
        # Compute expected G_s at perturbed T (using constant H_s and S_s as approximation)
        # In reality H_s and S_s are temperature-dependent, but for a small perturbation we assume constant.
        # This is a physics-based expectation: G decreases.
        perturbed_G = self.H_s - T_perturbed * self.S_s
        # The perturbed G should be less than baseline
        if perturbed_G >= baseline:
            return (False, f"Monotonicity violation: G_s at T+5% = {perturbed_G:.2f} not < baseline {baseline:.2f}")
        return (True, f"G_s decreases from {baseline:.2f} to {perturbed_G:.2f} when T increases 5%")

    # --------------------------------------------------------------------------
    # Weak answer sanity check (do not use isclose, do not re-derive final answer)
    # --------------------------------------------------------------------------
    def test_answer_sanity_broad_plausibility(self):
        """Weak checks: L positive, G negative, H_w positive and less than H_s."""
        if self.L is None or self.G_w is None or self.G_s is None or self.H_w is None:
            return (False, "SKIP: missing results for sanity check")
        issues = []
        if self.L <= 0:
            issues.append("L should be positive")
        if self.G_w >= 0:
            issues.append("G_w should be negative (phase equilibrium implies T*S > H for liquid)")
        if self.G_s >= 0:
            issues.append("G_s should be negative")
        if self.H_w <= 0:
            issues.append("H_w should be positive")
        if self.H_w >= self.H_s:
            issues.append("H_w should be less than H_s (heat absorbed during vaporization)")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All plausibility checks passed")

    # --------------------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------------------
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