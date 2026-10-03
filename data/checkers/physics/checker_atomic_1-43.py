import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits solutions for the Rydberg transition problem (n=109 → n=108)
    for hydrogen and helium.
    """

    # Known physical constants (SI + cm where appropriate)
    R_H = 1.097373e7        # m⁻¹
    c = 2.998e8             # m/s
    h = 6.626e-34           # J·s
    n_low = 108
    n_high = 109

    def __init__(self, solution_instance):
        # Extract results dict safely, NEVER numeric defaults
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.nu = results.get('nu', None)           # Hz
        self.lmbda = results.get('lambda', None)    # cm
        self.ratio = results.get('ratio', None)     # dimensionless
        self.difficulty = results.get('difficulty', None)  # string

        # Pre‑compute common denominator (exact value)
        self.delta = 1.0 / (self.n_low ** 2) - 1.0 / (self.n_high ** 2)

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four required result keys exist."""
        missing = []
        if self.nu is None:
            missing.append('nu')
        if self.lmbda is None:
            missing.append('lambda')
        if self.ratio is None:
            missing.append('ratio')
        if self.difficulty is None:
            missing.append('difficulty')
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present."

    def test_interface_nu_positive(self):
        if self.nu is None:
            return False, "SKIP: nu not available"
        if self.nu > 0:
            return True, f"nu = {self.nu:.3e} > 0"
        return False, f"nu = {self.nu:.3e} (non‑positive)"

    def test_interface_lambda_positive(self):
        if self.lmbda is None:
            return False, "SKIP: lambda not available"
        if self.lmbda > 0:
            return True, f"lambda = {self.lmbda:.3f} cm > 0"
        return False, f"lambda = {self.lmbda:.3f} cm (non‑positive)"

    def test_interface_ratio_positive(self):
        if self.ratio is None:
            return False, "SKIP: ratio not available"
        if self.ratio > 0:
            return True, f"ratio = {self.ratio:.3f} > 0"
        return False, f"ratio = {self.ratio:.3f} (non‑positive)"

    def test_interface_difficulty_is_string(self):
        if self.difficulty is None:
            return False, "SKIP: difficulty not available"
        if isinstance(self.difficulty, str) and len(self.difficulty) > 0:
            return True, "Difficulty is a non‑empty string."
        return False, f"Difficulty is {type(self.difficulty).__name__}, not a non‑empty string."

    # ------------------------------------------------------------
    # Invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_nu_rydberg(self):
        """ν should match Rydberg formula: ν = c·R_H·(1/108² − 1/109²)"""
        if self.nu is None:
            return False, "SKIP: nu not available"
        expected_nu = self.c * self.R_H * self.delta
        if math.isclose(self.nu, expected_nu, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"nu = {self.nu:.3e} Hz (expected {expected_nu:.3e} Hz)"
        return False, f"nu diverges: got {self.nu:.3e}, expected {expected_nu:.3e}"

    def test_invariant_lambda_rydberg(self):
        """λ (in cm) should match Rydberg formula: λ_cm = 100/(R_H·δ)"""
        if self.lmbda is None:
            return False, "SKIP: lambda not available"
        expected_lambda_cm = 100.0 / (self.R_H * self.delta)
        if math.isclose(self.lmbda, expected_lambda_cm, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"λ = {self.lmbda:.3f} cm (expected {expected_lambda_cm:.3f} cm)"
        return False, f"λ divergence: got {self.lmbda:.3f}, expected {expected_lambda_cm:.3f}"

    def test_invariant_nu_lambda_relation(self):
        """ν and λ should satisfy ν = c/λ (with λ in m)."""
        if self.nu is None or self.lmbda is None:
            return False, "SKIP: nu or lambda missing"
        lambda_m = self.lmbda / 100.0   # convert cm → m
        expected_nu = self.c / lambda_m
        if math.isclose(self.nu, expected_nu, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ν = {self.nu:.3e} Hz, c/λ = {expected_nu:.3e} Hz — consistent."
        return False, f"ν = {self.nu:.3e}, c/λ = {expected_nu:.3e} — mismatch."

    def test_invariant_energy_difference(self):
        """ΔE = hν and ΔE = R_H·h·c·δ should be consistent."""
        if self.nu is None:
            return False, "SKIP: nu not available"
        delta_E_expected = self.R_H * self.h * self.c * self.delta
        delta_E_from_nu = self.h * self.nu
        if math.isclose(delta_E_from_nu, delta_E_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ΔE = {delta_E_from_nu:.3e} J (expected {delta_E_expected:.3e} J)"
        return False, f"ΔE mismatch: {delta_E_from_nu:.3e} vs {delta_E_expected:.3e}"

    def test_invariant_ratio_approximately_one(self):
        """
        For neutral He (Z_eff ≈ 1) the ratio λ_He/λ_H ≈ 1.
        (This is the interpretation used in the explanation steps.)
        """
        if self.ratio is None:
            return False, "SKIP: ratio not available"
        # Allow ±10% tolerance (0.9 … 1.1)
        if math.isclose(self.ratio, 1.0, rel_tol=0.1):
            return True, f"ratio ≈ {self.ratio:.3f} (≈1) – consistent with neutral He."
        return False, f"ratio = {self.ratio:.3f}, expected ≈1 (Z_eff ≈ 1)"

    # ------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------
    def test_modeling_delta_conversion(self):
        """Check that δ = 1/108² − 1/109² was computed correctly."""
        delta_manual = 1.0 / 108**2 - 1.0 / 109**2
        if math.isclose(self.delta, delta_manual, rel_tol=1e-6):
            return True, f"δ = {self.delta:.6e} (expected {delta_manual:.6e})"
        return False, f"δ mismatch: {self.delta:.6e} vs {delta_manual:.6e}"

    def test_modeling_frequency_range(self):
        """ν should be in the reasonable range for n=109→108 (GHz)."""
        if self.nu is None:
            return False, "SKIP: nu not available"
        if 1e9 < self.nu < 1e10:   # typical range ~ 5 GHz
            return True, f"ν = {self.nu:.3e} Hz in GHz range."
        return False, f"ν = {self.nu:.3e} Hz outside 1–10 GHz."

    def test_trace_lambda_unit_conversion(self):
        """λ should be reported in cm: typical value ~ 6 cm, so a few cm."""
        if self.lmbda is None:
            return False, "SKIP: lambda not available"
        if 1.0 < self.lmbda < 20.0:
            return True, f"λ = {self.lmbda:.3f} cm (plausible cm value)."
        return False, f"λ = {self.lmbda:.3f} cm outside expected 1–20 cm."

    # ------------------------------------------------------------
    # Monotonicity check (perturb an input, compare output direction)
    # ------------------------------------------------------------
    def test_monotonicity_frequency_increases_with_lower_n_low(self):
        """
        If the lower level n_low is decreased (e.g., from 108 to 107),
        the energy difference ΔE increases, so the frequency ν should increase.
        We compare the solution's ν (baseline for n_low=108) with the
        ν' computed for n_low' = 107 (using the same Rydberg formula).
        """
        if self.nu is None:
            return False, "SKIP: nu not available"

        # Baseline: original n_low = 108
        baseline_nu = self.nu

        # Perturbed n_low (lower value → larger gap)
        perturbed_n_low = self.n_low - 1   # 107
        delta_prime = 1.0 / (perturbed_n_low ** 2) - 1.0 / (self.n_high ** 2)
        perturbed_nu = self.c * self.R_H * delta_prime   # derived from physics

        # Check monotonic direction: perturbed larger than baseline
        if perturbed_nu > baseline_nu:
            return True, (f"Perturbed ν (n_low={perturbed_n_low}) = {perturbed_nu:.3e} Hz"
                          f" > baseline ν = {baseline_nu:.3e} Hz — monotonicity holds.")
        else:
            return False, (f"Monotonicity violated: perturbed ν ({perturbed_nu:.3e})"
                           f" not larger than baseline ({baseline_nu:.3e}).")

    # ------------------------------------------------------------
    # Weak answer sanity (one method, secondary)
    # ------------------------------------------------------------
    def test_answer_sanity_difficulty(self):
        """
        Difficulty explanation should mention high‑n fragility and low
        transition probability (simple keyword check).
        """
        if self.difficulty is None:
            return False, "SKIP: difficulty not available"
        if not isinstance(self.difficulty, str):
            return False, "Difficulty is not a string."
        # Broad keywords – very permissive
        keywords = ["fragil", "ioniz", "low prob", "sensitive", "collision", "perturb"]
        found = any(kw in self.difficulty.lower() for kw in keywords)
        if found:
            return True, "Difficulty contains plausible physical reasoning."
        return False, f"Difficulty '{self.difficulty[:80]}' lacks expected keywords."

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs