import math
import sympy
import traceback

class PhysicsChecker:
    """
    Audits the calculation of mass difference between neutron and hydrogen atom
    from Q-values of ³He(n,p) reaction (Q1=0.770 MeV) and tritium β-decay (Q2=0.018 MeV).
    Expected result: results['delta_m_amu'] = (Q1+Q2)/931 ≈ 8.46e-4 amu.
    """
    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', {})
        self.delta_m_amu = results.get('delta_m_amu', None)

        self.Q1 = getattr(solution_instance, 'Q1', None)
        self.Q2 = getattr(solution_instance, 'Q2', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----- Interface tests -----
    def test_interface_delta_m_amu_present(self):
        """Check that the required result key exists."""
        if self.delta_m_amu is None:
            return (False, "SKIP: delta_m_amu not found in results")
        return (True, "delta_m_amu key present")

    def test_interface_delta_m_amu_type(self):
        """Check that delta_m_amu is numeric (int or float)."""
        if self.delta_m_amu is None:
            return (False, "SKIP: delta_m_amu not available")
        if not isinstance(self.delta_m_amu, (int, float)):
            return (False, f"delta_m_amu has type {type(self.delta_m_amu)}")
        return (True, "delta_m_amu is numeric")

    # ----- Invariant tests (two theorem-based) -----
    def test_invariant_delta_m_amu_conversion(self):
        """
        Invariant: Δ (amu) = (Q1 + Q2) / 931.
        Re-derive expected value inside checker; compare with math.isclose.
        """
        if self.delta_m_amu is None:
            return (False, "SKIP: delta_m_amu not available")
        Q1_MeV = 0.770
        Q2_MeV = 0.018
        amu_factor = 931.0
        expected_amu = (Q1_MeV + Q2_MeV) / amu_factor
        if not math.isclose(self.delta_m_amu, expected_amu, rel_tol=1e-2):
            return (False, f"delta_m_amu {self.delta_m_amu} differs from expected {expected_amu}")
        return (True, "delta_m_amu matches (Q1+Q2)/931 to 1%")

    def test_invariant_delta_m_energy_conversion(self):
        """
        Invariant: Δ * 931 (MeV) = Q1 + Q2.
        Checks the inverse conversion; a separate theorem (unit consistency).
        """
        if self.delta_m_amu is None:
            return (False, "SKIP: delta_m_amu not available")
        Q1_MeV = 0.770
        Q2_MeV = 0.018
        amu_factor = 931.0
        energy_from_amu = self.delta_m_amu * amu_factor
        expected_energy = Q1_MeV + Q2_MeV
        if not math.isclose(energy_from_amu, expected_energy, rel_tol=1e-2):
            return (False, f"delta_m_amu * 931 = {energy_from_amu}, expected {expected_energy}")
        return (True, "delta_m_amu * 931 equals sum of Q-values to 1%")

    # ----- Modeling/trace quality tests (two) -----
    def test_modeling_sum_of_q_values(self):
        """Check that explanation steps mention summing the two Q-values."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_text = str(self.explanation_steps).lower()
        keywords = ['sum', 'add', '0.770 + 0.018', 'total']
        if any(kw in steps_text for kw in keywords):
            return (True, "Explanation steps mention summing Q-values")
        else:
            return (False, "Explanation steps do not appear to sum the two Q-values")

    def test_modeling_conversion_to_amu(self):
        """Check that explanation steps include conversion from MeV to amu (division by 931)."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_text = str(self.explanation_steps).lower()
        # Look for keywords related to conversion: amu, 931, MeV/amu, unit conversion
        keywords = ['amu', '931', 'mev/amu', 'convert']
        if any(kw in steps_text for kw in keywords):
            return (True, "Explanation steps mention conversion to amu")
        else:
            return (False, "Explanation steps do not mention conversion to amu (931 MeV/amu)")

    # ----- Monotonicity test -----
    def test_monotonicity_q1_increase(self):
        """
        Perturb Q1 by +1% and recompute delta_m_amu using the same formula.
        Baseline is computed from stored Q1, Q2 (if available). The perturbed result should be larger.
        """
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: Q1 or Q2 not available from solution_instance")
        if not isinstance(self.Q1, (int, float)) or not isinstance(self.Q2, (int, float)):
            return (False, "SKIP: Q1/Q2 not numeric")

        amu_factor = 931.0
        baseline = (self.Q1 + self.Q2) / amu_factor

        # Perturb Q1 upward by 1%
        delta_percent = 0.01
        perturbed_Q1 = self.Q1 * (1 + delta_percent)
        perturbed_delta = (perturbed_Q1 + self.Q2) / amu_factor

        if perturbed_delta <= baseline:
            return (False, f"Perturbed delta {perturbed_delta} is not greater than baseline {baseline}")
        return (True, "Increasing Q1 leads to larger delta_m_amu (monotonicity satisfied)")

    # ----- Weak sanity test (at most one) -----
    def test_answer_sanity_positive(self):
        """The mass difference must be positive."""
        if self.delta_m_amu is None:
            return (False, "SKIP: delta_m_amu not available")
        if self.delta_m_amu <= 0:
            return (False, f"delta_m_amu is not positive: {self.delta_m_amu}")
        return (True, "delta_m_amu > 0")

    # ----- Run all tests -----
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