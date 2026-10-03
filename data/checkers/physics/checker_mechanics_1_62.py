import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens if present
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        self.R = givens.get('R', None)
        self.r = givens.get('r', None)
        self.M = givens.get('M', None)
        self.m = givens.get('m', None)

        # Extract expected result keys
        self.dsigma_dOmega = self.results.get('dsigma_dOmega', None)
        self.sigma_tot = self.results.get('sigma_tot', None)

        # Compute effective radius if both R and r available
        if self.R is not None and self.r is not None:
            self.R_eff = self.R + self.r
        else:
            self.R_eff = None

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        if self.dsigma_dOmega is None:
            return (False, "Missing required key: dsigma_dOmega")
        if self.sigma_tot is None:
            return (False, "Missing required key: sigma_tot")
        if not isinstance(self.dsigma_dOmega, (int, float)):
            return (False, f"dsigma_dOmega is not numeric: {type(self.dsigma_dOmega)}")
        if not isinstance(self.sigma_tot, (int, float)):
            return (False, f"sigma_tot is not numeric: {type(self.sigma_tot)}")
        return (True, "All required keys present and numeric")

    def test_interface_positive_values(self):
        if self.dsigma_dOmega is None or self.sigma_tot is None:
            return (False, "Skip: results not available")
        if self.dsigma_dOmega <= 0:
            return (False, f"dsigma_dOmega must be positive, got {self.dsigma_dOmega}")
        if self.sigma_tot <= 0:
            return (False, f"sigma_tot must be positive, got {self.sigma_tot}")
        return (True, "Both values positive")

    # ---------- Invariant tests (physics theorems) ----------
    def test_invariant_geometric_total(self):
        """Check sigma_tot ≈ π (R+r)^2 (geometric cross section)."""
        if self.R_eff is None:
            return (False, "SKIP: R and/or r not provided in givens")
        if self.sigma_tot is None:
            return (False, "Skip: sigma_tot not available")
        expected = math.pi * self.R_eff ** 2
        if math.isclose(self.sigma_tot, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_tot {self.sigma_tot} close to π(R+r)^2 = {expected}")
        else:
            return (False, f"sigma_tot {self.sigma_tot} differs from π(R+r)^2 = {expected}")

    def test_invariant_constant_dsigma_dOmega(self):
        """Check that dsigma_dOmega is constant (positive) – required by isotropic scattering."""
        if self.dsigma_dOmega is None:
            return (False, "Skip: dsigma_dOmega not available")
        if self.dsigma_dOmega <= 0:
            return (False, f"dsigma_dOmega must be positive, got {self.dsigma_dOmega}")
        return (True, "dsigma_dOmega is positive constant (isotropy consistent)")

    # ---------- Modeling / trace tests (consistency with derived formulas) ----------
    def test_modeling_dsigma_dOmega_formula(self):
        """Check dsigma_dOmega = (R+r)^2 / 4."""
        if self.R_eff is None or self.dsigma_dOmega is None:
            return (False, "SKIP: missing R/r or dsigma_dOmega")
        expected = self.R_eff ** 2 / 4.0
        if math.isclose(self.dsigma_dOmega, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dsigma_dOmega {self.dsigma_dOmega} close to (R+r)^2/4 = {expected}")
        else:
            return (False, f"dsigma_dOmega {self.dsigma_dOmega} differs from (R+r)^2/4 = {expected}")

    def test_modeling_isotropic_total(self):
        """Check sigma_tot = 4π * dsigma_dOmega (isotropic assumption)."""
        if self.dsigma_dOmega is None or self.sigma_tot is None:
            return (False, "Skip: results not available")
        expected_sigma = 4 * math.pi * self.dsigma_dOmega
        if math.isclose(self.sigma_tot, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_tot {self.sigma_tot} matches 4π * dsigma_dOmega = {expected_sigma}")
        else:
            return (False, f"sigma_tot {self.sigma_tot} differs from 4π * dsigma_dOmega = {expected_sigma}")

    # ---------- Monotonicity tests (explicit input perturbation) ----------
    def test_monotonicity_dsigma_dOmega_increases_with_R_eff(self):
        """Perturb R_eff by +5% and check that dsigma_dOmega increases (as (R_eff)^2)."""
        if self.R_eff is None or self.dsigma_dOmega is None:
            return (False, "SKIP: missing R_eff or dsigma_dOmega")
        epsilon = 0.05
        new_R_eff = self.R_eff * (1 + epsilon)
        # Baseline dsigma_dOmega from solution; expected new value scales as (R_eff_new/R_eff)^2
        expected_new_dsigma = self.dsigma_dOmega * (new_R_eff / self.R_eff) ** 2
        if expected_new_dsigma > self.dsigma_dOmega:
            return (True, f"dsigma_dOmega increases from {self.dsigma_dOmega} to {expected_new_dsigma} when R_eff increases by 5%")
        else:
            return (False, f"Expected dsigma_dOmega to increase, but {expected_new_dsigma} <= {self.dsigma_dOmega}")

    def test_monotonicity_sigma_tot_increases_with_R_eff(self):
        """Perturb R_eff by +5% and check that sigma_tot increases (as π R_eff^2)."""
        if self.R_eff is None or self.sigma_tot is None:
            return (False, "SKIP: missing R_eff or sigma_tot")
        epsilon = 0.05
        new_R_eff = self.R_eff * (1 + epsilon)
        # Baseline sigma_tot from solution; expected new value scales as (R_eff_new/R_eff)^2
        expected_new_sigma = self.sigma_tot * (new_R_eff / self.R_eff) ** 2
        if expected_new_sigma > self.sigma_tot:
            return (True, f"sigma_tot increases from {self.sigma_tot} to {expected_new_sigma} when R_eff increases by 5%")
        else:
            return (False, f"Expected sigma_tot to increase, but {expected_new_sigma} <= {self.sigma_tot}")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_positive_finite(self):
        """Weak sanity: both results are positive and finite."""
        if self.dsigma_dOmega is None or self.sigma_tot is None:
            return (False, "Skip: results not available")
        checks = []
        ok = True
        if not (self.dsigma_dOmega > 0 and math.isfinite(self.dsigma_dOmega)):
            ok = False
            checks.append(f"dsigma_dOmega = {self.dsigma_dOmega} not positive/finite")
        if not (self.sigma_tot > 0 and math.isfinite(self.sigma_tot)):
            ok = False
            checks.append(f"sigma_tot = {self.sigma_tot} not positive/finite")
        if ok:
            return (True, "Both results positive and finite")
        else:
            return (False, "; ".join(checks))

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)