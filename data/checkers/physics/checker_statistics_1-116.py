import math
import sys
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Physics Checker for the melting of ice problem.
    Checks interface completeness, invariant physics relations,
    modeling consistency, monotonicity, and weak answer sanity.
    """

    # Physical constants
    ATM_TO_PA = 101325.0
    CAL_TO_J = 4.184  # thermochemical calorie (1 cal = 4.184 J)
    G_PER_MOL = 18.0  # molar mass of water

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract input givens and results from solution_instance safely.
        All missing values are set to None.
        """
        # Input givens (expected)
        self.P_atm = getattr(solution_instance, 'P', None)  # in atm
        self.T_K = getattr(solution_instance, 'T', None)     # in Kelvin
        self.Q_kcal_per_mol = getattr(solution_instance, 'Q', None)  # in kcal/mol
        self.mass_g = getattr(solution_instance, 'mass', None)       # in grams (for 1 mole)
        self.rho_ice = getattr(solution_instance, 'rho_ice', None)   # g/cm³
        self.rho_water = getattr(solution_instance, 'rho_water', None) # g/cm³

        # Additional derived/conversion factors often present
        self.cal_to_erg = getattr(solution_instance, 'cal_to_erg', None)

        # Results (expected keys from pseudocode [emit] contract)
        results = getattr(solution_instance, 'results', {})
        self.W_cal = results.get('W_cal', None)
        self.U_change = results.get('U_change', None)
        self.S_change = results.get('S_change', None)

        # Explanation steps for reference (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: compute expected work in cal from givens (if all present)
    # ------------------------------------------------------------------
    def _compute_expected_W_cal(self) -> Optional[float]:
        """
        Re-derive work in calories from givens.
        Returns None if any required given is missing.
        Units: P in Pa, volumes in m³, work in J, then convert to cal.
        """
        if None in (self.P_atm, self.mass_g, self.rho_ice, self.rho_water):
            return None
        # Convert to SI
        P_pa = self.P_atm * self.ATM_TO_PA
        m_kg = self.mass_g / 1000.0
        rho_ice_kg_m3 = self.rho_ice * 1000.0  # g/cm³ -> kg/m³
        rho_water_kg_m3 = self.rho_water * 1000.0

        V_ice = m_kg / rho_ice_kg_m3
        V_water = m_kg / rho_water_kg_m3
        dV = V_water - V_ice   # negative because ice is less dense
        W_J = P_pa * dV       # work by system (negative)
        W_cal = W_J / self.CAL_TO_J
        return W_cal

    def _compute_expected_Q_cal(self) -> Optional[float]:
        """Convert Q from kcal/mol to cal for 1 mole."""
        if self.Q_kcal_per_mol is None:
            return None
        # 1 kcal = 1000 cal
        return self.Q_kcal_per_mol * 1000.0

    # ------------------------------------------------------------------
    # 1. Interface completeness tests
    # ------------------------------------------------------------------
    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all required input givens are present and positive."""
        required = [
            ("P_atm", self.P_atm),
            ("T_K", self.T_K),
            ("Q_kcal_per_mol", self.Q_kcal_per_mol),
            ("mass_g", self.mass_g),
            ("rho_ice", self.rho_ice),
            ("rho_water", self.rho_water),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            return (False, f"Missing input givens: {', '.join(missing)}")
        # Check positivity (temperatures above absolute zero, densities >0, mass>0, P>0)
        for name, val in required:
            if val <= 0:
                return (False, f"{name} is not positive (got {val})")
        # T must be >0 K, Q can be positive, etc.
        return (True, "All input givens present and positive.")

    def test_interface_results(self) -> Tuple[bool, str]:
        """Check that all expected result keys exist and are numeric."""
        required_results = [
            ("W_cal", self.W_cal),
            ("U_change", self.U_change),
            ("S_change", self.S_change),
        ]
        missing = [name for name, val in required_results if val is None]
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        for name, val in required_results:
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not a number (got {type(val)}).")
        return (True, "All result keys present and numeric.")

    # ------------------------------------------------------------------
    # 2. Invariant physics checks (theorem-based; use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_work_formula(self) -> Tuple[bool, str]:
        """Check that W_cal equals P ΔV (in cal) within tolerance."""
        if self.W_cal is None:
            return (False, "SKIP: W_cal not available")
        expected_W = self._compute_expected_W_cal()
        if expected_W is None:
            return (False, "SKIP: insufficient givens to compute expected work")
        # Use math.isclose with 1% relative tolerance (physics modeling)
        if math.isclose(self.W_cal, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Work formula holds (expected {expected_W:.4f} cal, got {self.W_cal:.4f})")
        else:
            return (False, f"Work mismatch: expected approx {expected_W:.4f} cal, got {self.W_cal:.4f}")

    def test_invariant_first_law(self) -> Tuple[bool, str]:
        """Check that ΔU = Q - W (sign convention: W is work by system)."""
        if None in (self.U_change, self.W_cal):
            return (False, "SKIP: U_change or W_cal missing")
        Q_cal = self._compute_expected_Q_cal()
        if Q_cal is None:
            return (False, "SKIP: Q input missing")
        expected_U = Q_cal - self.W_cal  # because W_cal already includes sign
        if math.isclose(self.U_change, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"First law holds (expected ΔU={expected_U:.2f} cal, got {self.U_change:.2f})")
        else:
            return (False, f"First law violation: expected ΔU={expected_U:.2f} cal, got {self.U_change:.2f}")

    def test_invariant_entropy_change(self) -> Tuple[bool, str]:
        """Check that ΔS = Q / T for reversible isothermal process."""
        if None in (self.S_change, self.T_K):
            return (False, "SKIP: S_change or T missing")
        Q_cal = self._compute_expected_Q_cal()
        if Q_cal is None:
            return (False, "SKIP: Q input missing")
        expected_S = Q_cal / self.T_K
        if math.isclose(self.S_change, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Entropy formula holds (expected {expected_S:.4f} cal/K, got {self.S_change:.4f})")
        else:
            return (False, f"Entropy mismatch: expected {expected_S:.4f} cal/K, got {self.S_change:.4f}")

    # ------------------------------------------------------------------
    # 3. Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_signs(self) -> Tuple[bool, str]:
        """Check physical sign conventions: W negative, ΔU positive, ΔS positive."""
        if None in (self.W_cal, self.U_change, self.S_change):
            return (False, "SKIP: missing result(s)")
        if self.W_cal >= 0:
            return (False, f"Work done by system should be negative (got {self.W_cal})")
        if self.U_change <= 0:
            return (False, f"Internal energy change should be positive (got {self.U_change})")
        if self.S_change <= 0:
            return (False, f"Entropy change should be positive (got {self.S_change})")
        # Also check that ΔU is greater than Q (since W is negative)
        Q_cal = self._compute_expected_Q_cal()
        if Q_cal is not None and self.U_change <= Q_cal:
            # Because W<0 => ΔU = Q - W > Q
            return (False, f"ΔU should be > Q (ΔU={self.U_change}, Q={Q_cal})")
        return (True, "All sign conventions satisfied.")

    def test_modeling_volume_change(self) -> Tuple[bool, str]:
        """Check that ΔV from explanation steps is negative using givens."""
        if None in (self.mass_g, self.rho_ice, self.rho_water):
            return (False, "SKIP: missing density/mass")
        V_ice = self.mass_g / self.rho_ice
        V_water = self.mass_g / self.rho_water
        dV = V_water - V_ice
        # dV should be negative (since ice is less dense)
        if dV >= 0:
            return (False, f"Volume change should be negative (dV={dV:.4f} cm³)")
        return (True, f"Volume change is negative: dV={dV:.4f} cm³")

    # ------------------------------------------------------------------
    # 4. Monotonicity tests (perturb one input, recompute effect)
    # ------------------------------------------------------------------
    def test_monotonicity_work_with_rho_ice(self) -> Tuple[bool, str]:
        """
        If ice density increases, the volume of ice decreases,
        so V_ice becomes smaller, |dV| decreases, work becomes less negative.
        """
        if not self._compute_expected_W_cal():
            return (False, "SKIP: cannot compute baseline work")
        # Baseline work
        W_base = self._compute_expected_W_cal()
        # Increase rho_ice by 1%
        rho_ice_new = self.rho_ice * 1.01
        # Temporarily modify self.rho_ice for calculation
        old_rho = self.rho_ice
        self.rho_ice = rho_ice_new
        W_perturbed = self._compute_expected_W_cal()
        # Restore
        self.rho_ice = old_rho
        if W_perturbed is None:
            return (False, "Error computing perturbed work")
        # Expect W_perturbed > W_base (less negative)
        if W_perturbed > W_base:
            return (True, f"Monotonicity holds: rho_ice +1% yields W = {W_perturbed:.6f} > {W_base:.6f}")
        else:
            return (False, f"Monotonicity violated: expected increase, got {W_perturbed:.6f} <= {W_base:.6f}")

    def test_monotonicity_work_with_pressure(self) -> Tuple[bool, str]:
        """If pressure increases, the magnitude of work increases (more negative)."""
        if not self._compute_expected_W_cal():
            return (False, "SKIP: cannot compute baseline work")
        W_base = self._compute_expected_W_cal()
        # Increase pressure by 1%
        P_new = self.P_atm * 1.01
        old_P = self.P_atm
        self.P_atm = P_new
        W_perturbed = self._compute_expected_W_cal()
        self.P_atm = old_P
        if W_perturbed is None:
            return (False, "Error computing perturbed work")
        # Expect W_perturbed < W_base (more negative)
        if W_perturbed < W_base:
            return (True, f"Monotonicity holds: P +1% yields W = {W_perturbed:.6f} < {W_base:.6f}")
        else:
            return (False, f"Monotonicity violated: expected decrease, got {W_perturbed:.6f} >= {W_base:.6f}")

    # ------------------------------------------------------------------
    # 5. Answer sanity (weak checks only, no re-derivation with isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """
        Very weak sanity: W_cal ~ -0.034 cal, U_change ~ 1436.3 cal,
        S_change ~ 5.26 cal/K. Only check within broad order-of-magnitude range.
        """
        if None in (self.W_cal, self.U_change, self.S_change):
            return (False, "SKIP: missing results")
        # Work: expect negative, magnitude in range [0.001, 1] cal
        if not (-1.0 < self.W_cal < -0.001):
            return (False, f"W_cal magnitude outside expected range ({self.W_cal})")
        # ΔU: expect ~1436.3 cal, allow ±100 cal
        if not (1300.0 < self.U_change < 1600.0):
            return (False, f"U_change magnitude outside expected range ({self.U_change})")
        # ΔS: expect ~5.26 cal/K, allow ±1
        if not (4.0 < self.S_change < 7.0):
            return (False, f"S_change magnitude outside expected range ({self.S_change})")
        return (True, "Answer magnitudes within broad expected ranges.")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and execute all test methods.
        Returns (all_pass, logs).
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
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)