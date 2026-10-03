import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the Σ* hyperon problem.
    Checks strangeness, decay type, and isospin assignments using conservation laws.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.S_Sigma = results.get('S_Sigma', None)
        self.decay_type = results.get('decay_type', None)
        self.I3_Sigma = results.get('I3_Sigma', None)
        self.I_Sigma = results.get('I_Sigma', None)

        # Extract givens (if available)
        self.mass = getattr(solution_instance, 'm_Sigma', None) or getattr(solution_instance, 'mass', None)
        self.Gamma = getattr(solution_instance, 'Gamma', None) or getattr(solution_instance, 'width', None)
        self.BR = getattr(solution_instance, 'BR', None) or getattr(solution_instance, 'branching_ratio', None)

        # Strangeness and isospin constants (standard values)
        self.S_Km = -1
        self.S_Kp = 1
        self.S_p = 0
        self.S_pi = 0
        self.I3_Km = -0.5
        self.I3_Kp = 0.5
        self.I3_p = 0.5
        self.I3_pi_minus = -1
        self.I3_pi_plus = 1
        self.I3_Lambda = 0

        # Physical constant
        self.hbar = 6.582119569e-22  # MeV·s

    # ---- Interface checks ----
    def test_interface_S_Sigma_key(self):
        if self.S_Sigma is None:
            return (False, "SKIP: S_Sigma not available in results")
        return (True, f"S_Sigma key present with value {self.S_Sigma}")

    def test_interface_decay_type_key(self):
        if self.decay_type is None:
            return (False, "SKIP: decay_type not available in results")
        return (True, f"decay_type key present with value {self.decay_type}")

    def test_interface_I3_Sigma_key(self):
        if self.I3_Sigma is None:
            return (False, "SKIP: I3_Sigma not available in results")
        return (True, f"I3_Sigma key present with value {self.I3_Sigma}")

    def test_interface_I_Sigma_key(self):
        if self.I_Sigma is None:
            return (False, "SKIP: I_Sigma not available in results")
        return (True, f"I_Sigma key present with value {self.I_Sigma}")

    # ---- Invariant / theorem checks ----
    def test_invariant_strangeness_conservation(self):
        """Check strangeness conservation in K⁻ p → π⁻ Σ*⁺"""
        if self.S_Sigma is None:
            return (False, "SKIP: S_Sigma missing")
        initial_S = self.S_Km + self.S_p   # -1 + 0 = -1
        final_S = self.S_pi + self.S_Sigma  # 0 + S_Sigma
        if math.isclose(initial_S, final_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Strangeness conserved: initial {initial_S} = final {final_S}")
        else:
            return (False, f"Strangeness not conserved: initial {initial_S} != final {final_S}")

    def test_invariant_isospin_production(self):
        """Check I3 conservation in K⁻ p → π⁻ Σ*⁺"""
        if self.I3_Sigma is None:
            return (False, "SKIP: I3_Sigma missing")
        initial_I3 = self.I3_Km + self.I3_p   # -0.5 + 0.5 = 0
        final_I3 = self.I3_pi_minus + self.I3_Sigma  # -1 + I3_Sigma
        if math.isclose(initial_I3, final_I3, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I3 conserved in production: initial {initial_I3} = final {final_I3}")
        else:
            return (False, f"I3 not conserved: initial {initial_I3} != final {final_I3}")

    def test_invariant_width_scale(self):
        """Check that Gamma (if present) is of order MeV consistent with strong decay"""
        if self.Gamma is None:
            return (False, "SKIP: Gamma missing")
        if self.Gamma > 1.0:
            return (True, f"Gamma = {self.Gamma} MeV >> 1 MeV, consistent with strong decay")
        else:
            return (False, f"Gamma = {self.Gamma} MeV is too small for strong decay")

    def test_invariant_decay_consistency(self):
        """If decay_type is 'strong', check Gamma > 1 MeV; if 'weak', check Gamma < 1e-10 MeV"""
        if self.decay_type is None or self.Gamma is None:
            return (False, "SKIP: decay_type or Gamma missing")
        if self.decay_type == 'strong':
            if self.Gamma > 1.0:
                return (True, "strong decay matches large width")
            else:
                return (False, "strong decay label but width too small")
        elif self.decay_type == 'weak':
            if self.Gamma < 1e-10:
                return (True, "weak decay matches tiny width")
            else:
                return (False, "weak decay label but width too large")
        else:
            return (False, f"decay_type '{self.decay_type}' not recognized")

    # ---- Modeling / trace checks ----
    def test_modeling_isospin_from_decay(self):
        """Check that I3_Sigma matches I3 conservation in decay Σ*⁺ → π⁺ Λ"""
        if self.I3_Sigma is None:
            return (False, "SKIP: I3_Sigma missing")
        expected_I3 = self.I3_pi_plus + self.I3_Lambda   # 1
        if math.isclose(self.I3_Sigma, expected_I3, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I3_Sigma = {self.I3_Sigma} matches decay conservation")
        else:
            return (False, f"I3_Sigma = {self.I3_Sigma} does not equal expected {expected_I3} from decay")

    def test_modeling_total_isospin_range(self):
        """Check that I_Sigma >= |I3_Sigma| and I_Sigma is (half-)integer"""
        if self.I_Sigma is None or self.I3_Sigma is None:
            return (False, "SKIP: I_Sigma or I3_Sigma missing")
        if self.I_Sigma < abs(self.I3_Sigma):
            return (False, f"I_Sigma = {self.I_Sigma} < |I3_Sigma| = {abs(self.I3_Sigma)}")
        if not math.isclose(2 * self.I_Sigma, round(2 * self.I_Sigma), rel_tol=1e-2):
            return (False, f"I_Sigma = {self.I_Sigma} not a half-integer")
        return (True, f"I_Sigma = {self.I_Sigma} >= |I3_Sigma| = {abs(self.I3_Sigma)} and is half-integer")

    def test_modeling_charge_vs_strangeness(self):
        """Q = I3 + (B+S)/2 consistency check (Gell-Mann–Nishijima)"""
        if None in (self.S_Sigma, self.I3_Sigma):
            return (False, "SKIP: S_Sigma or I3_Sigma missing")
        B = 1
        Q = 1  # Σ*+ has charge +1
        expected_I3 = Q - (B + self.S_Sigma) / 2.0
        if math.isclose(self.I3_Sigma, expected_I3, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Gell-Mann–Nishijima satisfied")
        else:
            return (False, f"I3 = {self.I3_Sigma} but predicted {expected_I3}")

    # ---- Monotonicity checks ----
    def test_monotonicity_lifetime_vs_width(self):
        """
        If Gamma increases, the lifetime τ = ħ/Γ should decrease.
        Perturb Gamma by +10% and check that τ decreases.
        """
        if self.Gamma is None:
            return (False, "SKIP: Gamma missing for monotonicity check")
        baseline_tau = self.hbar / self.Gamma
        perturbed_Gamma = self.Gamma * 1.1  # increase by 10%
        perturbed_tau = self.hbar / perturbed_Gamma
        if perturbed_tau < baseline_tau:
            return (True, f"Increasing Gamma decreases tau: baseline {baseline_tau:e} -> perturbed {perturbed_tau:e}")
        else:
            return (False, f"tau did not decrease as expected: baseline {baseline_tau:e}, perturbed {perturbed_tau:e}")

    def test_monotonicity_partial_width_vs_branching_ratio(self):
        """
        If branching ratio increases, the partial width Γ_πΛ = BR * Γ should increase.
        Perturb BR by +10% (if BR is available) and verify partial width increases.
        """
        if self.Gamma is None or self.BR is None:
            return (False, "SKIP: Gamma or BR missing for monotonicity check")
        baseline_partial = self.BR * self.Gamma
        perturbed_BR = self.BR * 1.1  # increase by 10%
        perturbed_partial = perturbed_BR * self.Gamma
        if perturbed_partial > baseline_partial:
            return (True, f"Increasing BR increases partial width: baseline {baseline_partial} -> perturbed {perturbed_partial}")
        else:
            return (False, f"Partial width did not increase: baseline {baseline_partial}, perturbed {perturbed_partial}")

    # ---- Weak answer sanity (exactly one method) ----
    def test_answer_sanity_basic(self):
        """
        Weak sanity: S_Sigma should be integer, I_Sigma positive, decay_type string.
        No re-derivation, no isclose.
        """
        if self.S_Sigma is None or self.I_Sigma is None or self.decay_type is None:
            return (False, "SKIP: one or more result values missing")
        if not isinstance(self.S_Sigma, int):
            return (False, f"S_Sigma={self.S_Sigma} is not integer")
        if not isinstance(self.I_Sigma, (int, float)) or self.I_Sigma <= 0:
            return (False, f"I_Sigma={self.I_Sigma} not positive numeric")
        if not isinstance(self.decay_type, str) or len(self.decay_type) == 0:
            return (False, f"decay_type={self.decay_type} not non-empty string")
        return (True, "Basic sanity passed: S_Sigma integer, I_Sigma positive, decay_type string")

    # ---- Run all tests ----
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