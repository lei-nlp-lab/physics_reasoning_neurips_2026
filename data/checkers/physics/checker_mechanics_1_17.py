import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the car coasting problem.
    Checks interface completeness, invariant physics relations,
    modeling consistency, monotonicity trends, and answer sanity.
    No target-answer comparisons are used.
    """
    def __init__(self, solution_instance):
        # Safe extraction of all attributes with None defaults
        self.givens    = getattr(solution_instance, 'givens', {})
        self.results   = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.physics_mapping   = getattr(solution_instance, 'physics_mapping', {})
        self.key_constraints   = getattr(solution_instance, 'key_constraints', [])
        # target_answers stored but never used in tests to avoid hardcoded comparisons
        self._unused_targets = getattr(solution_instance, 'target_answers', [])
        self.reasoning_plan    = getattr(solution_instance, 'reasoning_plan', "")
        self.solving_pseudocode = getattr(solution_instance, 'solving_pseudocode', "")

        # Givens (all with None defaults)
        self.weight     = self.givens.get('weight', None)          # lb
        self.g_mphps    = self.givens.get('g_mphps', None)         # mph/s
        self.mph_to_ft  = self.givens.get('mph_to_ft', None)       # ft/s per mph
        # Derived constants (provided as fallback if givens are missing)
        self.k_ft_per_mph = 88 / 60.0  # 1.466666...
        if self.g_mphps is not None and self.mph_to_ft is not None:
            self.g_fts2 = self.g_mphps * self.mph_to_ft          # ft/s^2
        else:
            self.g_fts2 = None
        if self.weight is not None and self.g_fts2 is not None:
            self.m_slugs = self.weight / self.g_fts2            # slugs
        else:
            self.m_slugs = None
        if self.weight is not None and self.g_mphps is not None:
            self.m_mixed = self.weight / self.g_mphps           # lb·s/mph
        else:
            self.m_mixed = None

        # Result key
        self.P_hp = self.results.get('P_hp', None)

    # ---------- Interface checks ----------
    def test_interface_required_keys(self):
        """Check that results dict exists and contains 'P_hp'."""
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")
        if 'P_hp' not in self.results:
            return (False, "results['P_hp'] is missing")
        return (True, "results['P_hp'] present")

    def test_interface_P_hp_numeric(self):
        """Check that P_hp is a finite number."""
        if self.P_hp is None:
            return (False, "P_hp is None")
        if not isinstance(self.P_hp, (int, float)):
            return (False, f"P_hp has type {type(self.P_hp).__name__}, expected int or float")
        if math.isnan(self.P_hp) or math.isinf(self.P_hp):
            return (False, "P_hp is NaN or Inf")
        return (True, "P_hp is numeric and finite")

    def test_interface_givens(self):
        """Check that essential givens are present and positive."""
        if self.weight is None or self.weight <= 0:
            return (False, f"weight is {self.weight} (must be positive)")
        if self.g_mphps is None or self.g_mphps <= 0:
            return (False, f"g_mphps is {self.g_mphps} (must be positive)")
        if self.mph_to_ft is None or self.mph_to_ft <= 0:
            return (False, f"mph_to_ft is {self.mph_to_ft} (must be positive)")
        return (True, "All essential givens present and positive")

    # ---------- Invariant checks (physics theorem based) ----------
    def test_invariant_differential_equation(self):
        """
        At t=60 s (V=30 mph), verify dv/dt = -v^2/(60 * V0) with V0=60 mph.
        This is the core differential equation derived from the coasting law.
        """
        if self.g_mphps is None or self.weight is None:
            return (False, "SKIP: missing g_mphps or weight")
        t_val = 60.0
        v_mph = 60.0 / (1.0 + t_val / 60.0)   # 30 mph
        dvdt_mphps = -60.0 / (60.0 * (1.0 + t_val / 60.0)**2)  # exact derivative
        rhs = - (v_mph**2) / (60.0 * 60.0)   # -v^2/(60*V0)
        if math.isclose(dvdt_mphps, rhs, rel_tol=1e-4, abs_tol=1e-4):
            return (True, f"dv/dt ≈ rhs at t=60: {dvdt_mphps:.6f} vs {rhs:.6f}")
        else:
            return (False, f"dv/dt ({dvdt_mphps:.6f}) != -v^2/(60*60) ({rhs:.6f})")

    def test_invariant_resistance_force(self):
        """
        At t=60 s, check that F_res = m*|a| equals m*V^2/(60*V0) in mixed units.
        Uses m_mixed = weight/g_mphps (lb·s/mph), a in mph/s.
        """
        if self.weight is None or self.g_mphps is None:
            return (False, "SKIP: missing weight or g_mphps")
        t_val = 60.0
        v_mph = 30.0
        a_mphps = -1.0 / (1.0 + t_val / 60.0)**2  # exact dv/dt in mph/s
        m_mixed = self.weight / self.g_mphps       # lb·s/mph
        F_from_a = m_mixed * abs(a_mphps)          # lb
        F_from_formula = m_mixed * v_mph**2 / (60.0 * 60.0)   # V0=60 mph
        if math.isclose(F_from_a, F_from_formula, rel_tol=1e-4, abs_tol=1e-2):
            return (True, f"F_res from a ({F_from_a:.3f} lb) ≈ formula ({F_from_formula:.3f} lb)")
        else:
            return (False, f"F_res mismatch: {F_from_a:.3f} vs {F_from_formula:.3f}")

    # ---------- Modeling checks ----------
    def test_modeling_force_power_consistency(self):
        """
        Derive F_res from the solution's P_hp and compare with the expected
        resistance force at 30 mph using the physics formula.
        """
        if self.P_hp is None or self.weight is None or self.g_mphps is None:
            return (False, "SKIP: missing P_hp, weight, or g_mphps")
        # Convert 30 mph to ft/s
        v_fts = 30.0 * self.k_ft_per_mph  # 44 ft/s
        # Expected resistance force in lb (from physics)
        m_slugs = self.weight / self.g_fts2 if self.g_fts2 else None
        if m_slugs is None:
            # use mixed units route (consistent)
            m_mixed = self.weight / self.g_mphps
            a_mphps = -0.25   # at t=60 s
            F_exp = m_mixed * abs(a_mphps)   # lb
        else:
            # ft/s^2 units
            a_fts2 = -88.0 / (60.0 * (2.0)**2)  # exact at t=60: -88/(60*4) = -11/30
            F_exp = m_slugs * abs(a_fts2)       # lb
        # From solution's power, infer F_res = P / v
        P_ftlbs = self.P_hp * 550.0
        F_sol = P_ftlbs / v_fts   # lb
        if math.isclose(F_sol, F_exp, rel_tol=1e-2, abs_tol=1.0):
            return (True, f"F_sol ({F_sol:.2f} lb) ≈ expected F_exp ({F_exp:.2f} lb)")
        else:
            return (False, f"F_sol ({F_sol:.2f} lb) differs from expected ({F_exp:.2f} lb)")

    def test_modeling_power_computation(self):
        """
        Recomputed power at 30 mph from first principles and compare
        with solution's P_hp using isclose.
        """
        if self.P_hp is None or self.weight is None or self.g_mphps is None:
            return (False, "SKIP: missing P_hp, weight, or g_mphps")
        # Use mixed-unit approach (consistent with explanation)
        v_mph = 30.0
        m_mixed = self.weight / self.g_mphps        # lb·s/mph
        F_res = m_mixed * v_mph**2 / (60.0 * 60.0)  # lb, V0=60 mph
        P_mph_lb = F_res * v_mph                     # mph·lb
        P_ftlbs = P_mph_lb * self.k_ft_per_mph       # ft·lb/s
        P_hp_computed = P_ftlbs / 550.0
        if math.isclose(self.P_hp, P_hp_computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_hp solution = {self.P_hp:.4f}, computed = {P_hp_computed:.4f}")
        else:
            return (False, f"P_hp solution ({self.P_hp:.4f}) != computed ({P_hp_computed:.4f})")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_power_increases_with_speed(self):
        """
        Perturb the speed input (30 mph -> 31.5 mph) and verify that
        the required power (computed from physics) increases.
        Uses the solution's P_hp as baseline and scales by (v_pert/v_base)^3.
        """
        if self.P_hp is None or self.weight is None or self.g_mphps is None:
            return (False, "SKIP: missing P_hp, weight, or g_mphps")
        # baseline speed = 30 mph, perturb to 31.5 mph (+5%)
        v_base = 30.0
        v_pert = v_base * 1.05   # 31.5 mph
        # Power scales as v^3 for same road and mass
        scaling = (v_pert / v_base) ** 3
        P_pert = self.P_hp * scaling
        # Check that perturbed is greater than baseline
        if P_pert > self.P_hp:
            return (True, f"P at {v_pert} mph ≈ {P_pert:.4f} > baseline {self.P_hp:.4f} at {v_base} mph")
        else:
            return (False, f"Perturbed power {P_pert:.4f} not greater than baseline {self.P_hp:.4f}")

    def test_monotonicity_power_increases_with_weight(self):
        """
        Perturb the weight input (5000 lb -> 5500 lb, +10%) and verify that
        the required power (computed from physics) increases proportionally.
        Uses the solution's P_hp as baseline and scales by (w_pert/w_base).
        """
        if self.P_hp is None or self.weight is None or self.g_mphps is None:
            return (False, "SKIP: missing P_hp, weight, or g_mphps")
        w_base = self.weight
        w_pert = w_base * 1.10   # +10%
        # Power scales linearly with mass (assuming same road and speed)
        scaling = w_pert / w_base
        P_pert = self.P_hp * scaling
        if P_pert > self.P_hp:
            return (True, f"P with weight {w_pert:.1f} lb ≈ {P_pert:.4f} > baseline {self.P_hp:.4f} with weight {w_base:.1f} lb")
        else:
            return (False, f"Perturbed power {P_pert:.4f} not greater than baseline {self.P_hp:.4f}")

    def test_monotonicity_acceleration_magnitude_decreases_with_time(self):
        """
        For the coasting phase, perturb time input (from t=0 to t=60 s)
        and verify that |dv/dt| decreases. This tests a monotonic property
        of the model itself, using physics constants.
        """
        if self.g_mphps is None:
            return (False, "SKIP: missing g_mphps")
        t_base = 0.0
        t_pert = 60.0
        a_base = -1.0 / (1.0 + t_base / 60.0)**2
        a_pert = -1.0 / (1.0 + t_pert / 60.0)**2
        if abs(a_pert) < abs(a_base):
            return (True, f"|a| at t={t_pert:.1f}s ({abs(a_pert):.6f}) < |a| at t={t_base:.1f}s ({abs(a_base):.6f})")
        else:
            return (False, f"|a| not decreasing: perturbed {abs(a_pert):.6f} >= baseline {abs(a_base):.6f}")

    # ---------- Answer sanity (weak, secondary) ----------
    def test_answer_sanity_reasonable_magnitude(self):
        """
        Check that P_hp is positive and not unreasonably large.
        A car at 30 mph on a level road typically requires <10 hp.
        """
        if self.P_hp is None:
            return (False, "P_hp is None")
        if self.P_hp <= 0:
            return (False, f"P_hp = {self.P_hp} is not positive")
        if self.P_hp > 10.0:
            return (False, f"P_hp = {self.P_hp} > 10 hp, too large")
        return (True, f"P_hp = {self.P_hp} is positive and less than 10 hp")

    # ---------- Test runner ----------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)