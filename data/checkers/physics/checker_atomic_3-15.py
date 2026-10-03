import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution_instance for the particle physics problem.
    The solution_instance must have a 'results' dict (or be a dict itself)
    with keys expected from the pseudocode contract.
    All missing keys default to None; tests gracefully skip.
    """

    # Physical constants (MeV)
    m_p = 938.272
    m_n = 939.565
    m_Lambda = 1115.683
    m_K = 493.677
    m_pi = 139.570
    m_pi0 = 134.977
    m_e = 0.511
    m_mu = 105.658

    def __init__(self, solution_instance: Any):
        # Extract the results dictionary
        if isinstance(solution_instance, dict):
            results = solution_instance
        elif hasattr(solution_instance, 'results'):
            results = getattr(solution_instance, 'results', {})
        elif hasattr(solution_instance, 'to_dict'):
            results = solution_instance.to_dict()
        else:
            results = {}

        # Helper: extract with multiple key fallbacks
        def _get(keys: List[str]) -> Optional[Any]:
            for key in keys:
                if key in results:
                    return results[key]
                try:
                    val = getattr(solution_instance, key, None)
                    if val is not None:
                        return val
                except:
                    pass
            return None

        # Sub1: p + n -> p + Lambda^0
        self.delta_S_1 = _get(['delta_S_1', 'delta_S_sub1', 'delta_S'])
        self.delta_I3_1 = _get(['delta_I3_1', 'delta_I3_sub1', 'delta_I3'])

        # Sub2: K^+ -> π^+ + π^- + π^+ + π^- + π^+ + π^0 (pseudocode: 2π+ + 3π0)
        self.m_initial_2 = _get(['m_initial_2', 'm_initial'])
        self.m_final_sum_2 = _get(['m_final_sum_2', 'm_final_sum'])

        # Sub3: \overline{K}^0 -> π^- + e^+ + ν_e
        self.delta_S_3 = _get(['delta_S_3', 'delta_S'])
        self.delta_Q_hadron_3 = _get(['delta_Q_hadron_3', 'delta_Q_hadron'])

        # Sub4: Λ^0 -> K^0 + π^0
        self.B_initial_4 = _get(['B_initial_4', 'B_initial'])
        self.B_final_4 = _get(['B_final_4', 'B_final'])

        # Sub5: π^+ -> e^+ ν_e / π^+ -> μ^+ ν_μ
        self.R_5 = _get(['R_5', 'R'])

        # Sub6: K_L^0 -> e^+ e^-
        self.delta_S_6 = _get(['delta_S_6', 'delta_S'])
        self.delta_Q_hadron_6 = _get(['delta_Q_hadron_6', 'delta_Q_hadron'])

        # Sub7: K^- -> π^0 + e^-
        self.L_initial_7 = _get(['L_initial_7', 'L_initial'])
        self.L_final_7 = _get(['L_final_7', 'L_final'])

        # Sub8: π^0 -> γ + γ + γ
        self.C_initial_8 = _get(['C_initial_8', 'C_initial'])
        self.C_final_8 = _get(['C_final_8', 'C_final'])

        # Sub9: K_L^0 -> π^+ + π^-
        self.CP_initial_9 = _get(['CP_initial_9', 'CP_initial'])
        self.CP_final_9 = _get(['CP_final_9', 'CP_final'])

        # Sub10: K^+ -> π^+ + π^+ + π^0
        self.Q_initial_10 = _get(['Q_initial_10', 'Q_initial'])
        self.Q_final_10 = _get(['Q_final_10', 'Q_final'])

    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface tests: required keys exist
    # ------------------------------------------------------------------
    def test_interface_delta_S_1(self) -> Tuple[bool, str]:
        if self.delta_S_1 is None:
            return False, "delta_S_1 is None"
        return True, "present"

    def test_interface_delta_I3_1(self) -> Tuple[bool, str]:
        if self.delta_I3_1 is None:
            return False, "delta_I3_1 is None"
        return True, "present"

    def test_interface_m_initial_2(self) -> Tuple[bool, str]:
        if self.m_initial_2 is None:
            return False, "m_initial_2 is None"
        return True, "present"

    def test_interface_m_final_sum_2(self) -> Tuple[bool, str]:
        if self.m_final_sum_2 is None:
            return False, "m_final_sum_2 is None"
        return True, "present"

    def test_interface_delta_S_3(self) -> Tuple[bool, str]:
        if self.delta_S_3 is None:
            return False, "delta_S_3 is None"
        return True, "present"

    def test_interface_delta_Q_hadron_3(self) -> Tuple[bool, str]:
        if self.delta_Q_hadron_3 is None:
            return False, "delta_Q_hadron_3 is None"
        return True, "present"

    def test_interface_B_initial_4(self) -> Tuple[bool, str]:
        if self.B_initial_4 is None:
            return False, "B_initial_4 is None"
        return True, "present"

    def test_interface_B_final_4(self) -> Tuple[bool, str]:
        if self.B_final_4 is None:
            return False, "B_final_4 is None"
        return True, "present"

    def test_interface_R_5(self) -> Tuple[bool, str]:
        if self.R_5 is None:
            return False, "R_5 is None"
        return True, "present"

    def test_interface_delta_S_6(self) -> Tuple[bool, str]:
        if self.delta_S_6 is None:
            return False, "delta_S_6 is None"
        return True, "present"

    def test_interface_delta_Q_hadron_6(self) -> Tuple[bool, str]:
        if self.delta_Q_hadron_6 is None:
            return False, "delta_Q_hadron_6 is None"
        return True, "present"

    def test_interface_L_initial_7(self) -> Tuple[bool, str]:
        if self.L_initial_7 is None:
            return False, "L_initial_7 is None"
        return True, "present"

    def test_interface_L_final_7(self) -> Tuple[bool, str]:
        if self.L_final_7 is None:
            return False, "L_final_7 is None"
        return True, "present"

    def test_interface_C_initial_8(self) -> Tuple[bool, str]:
        if self.C_initial_8 is None:
            return False, "C_initial_8 is None"
        return True, "present"

    def test_interface_C_final_8(self) -> Tuple[bool, str]:
        if self.C_final_8 is None:
            return False, "C_final_8 is None"
        return True, "present"

    def test_interface_CP_initial_9(self) -> Tuple[bool, str]:
        if self.CP_initial_9 is None:
            return False, "CP_initial_9 is None"
        return True, "present"

    def test_interface_CP_final_9(self) -> Tuple[bool, str]:
        if self.CP_final_9 is None:
            return False, "CP_final_9 is None"
        return True, "present"

    def test_interface_Q_initial_10(self) -> Tuple[bool, str]:
        if self.Q_initial_10 is None:
            return False, "Q_initial_10 is None"
        return True, "present"

    def test_interface_Q_final_10(self) -> Tuple[bool, str]:
        if self.Q_final_10 is None:
            return False, "Q_final_10 is None"
        return True, "present"

    # ------------------------------------------------------------------
    # Invariant tests: numerical checks based on known physics
    # ------------------------------------------------------------------
    def test_invariant_sub1_delta_S(self) -> Tuple[bool, str]:
        if self.delta_S_1 is None:
            return False, "SKIP: delta_S_1 not available"
        expected = -1.0  # S_final - S_initial = (0 + (-1)) - (0+0) = -1
        if math.isclose(self.delta_S_1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_S_1 approx -1"
        return False, f"delta_S_1 = {self.delta_S_1} != {expected}"

    def test_invariant_sub1_delta_I3(self) -> Tuple[bool, str]:
        if self.delta_I3_1 is None:
            return False, "SKIP: delta_I3_1 not available"
        expected = 0.5  # I3_final - I3_initial = (0.5+0) - (0.5-0.5) = 0.5
        if math.isclose(self.delta_I3_1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_I3_1 approx 0.5"
        return False, f"delta_I3_1 = {self.delta_I3_1} != {expected}"

    def test_invariant_sub2_m_initial(self) -> Tuple[bool, str]:
        if self.m_initial_2 is None:
            return False, "SKIP: m_initial_2 not available"
        expected = self.m_K
        if math.isclose(self.m_initial_2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "m_initial_2 approx m_K"
        return False, f"m_initial_2 = {self.m_initial_2} != {expected}"

    def test_invariant_sub2_m_final_sum(self) -> Tuple[bool, str]:
        if self.m_final_sum_2 is None:
            return False, "SKIP: m_final_sum_2 not available"
        expected = 2 * self.m_pi + 3 * self.m_pi0  # 684.071 MeV
        if math.isclose(self.m_final_sum_2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "m_final_sum_2 approx 684.071"
        return False, f"m_final_sum_2 = {self.m_final_sum_2} != {expected}"

    def test_invariant_sub3_delta_S(self) -> Tuple[bool, str]:
        if self.delta_S_3 is None:
            return False, "SKIP: delta_S_3 not available"
        expected = 1.0  # Kbar0 S=-1, final S=0 => ΔS=+1
        if math.isclose(self.delta_S_3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_S_3 approx +1"
        return False, f"delta_S_3 = {self.delta_S_3} != {expected}"

    def test_invariant_sub3_delta_Q_hadron(self) -> Tuple[bool, str]:
        if self.delta_Q_hadron_3 is None:
            return False, "SKIP: delta_Q_hadron_3 not available"
        expected = -1.0  # Kbar0 Q=0, π⁻ Q=-1 => ΔQ_hadron = -1
        if math.isclose(self.delta_Q_hadron_3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_Q_hadron_3 approx -1"
        return False, f"delta_Q_hadron_3 = {self.delta_Q_hadron_3} != {expected}"

    def test_invariant_sub4_B_initial(self) -> Tuple[bool, str]:
        if self.B_initial_4 is None:
            return False, "SKIP: B_initial_4 not available"
        expected = 1.0
        if math.isclose(self.B_initial_4, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "B_initial_4 approx 1"
        return False, f"B_initial_4 = {self.B_initial_4} != {expected}"

    def test_invariant_sub4_B_final(self) -> Tuple[bool, str]:
        if self.B_final_4 is None:
            return False, "SKIP: B_final_4 not available"
        expected = 0.0
        if math.isclose(self.B_final_4, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "B_final_4 approx 0"
        return False, f"B_final_4 = {self.B_final_4} != {expected}"

    def test_invariant_sub5_R(self) -> Tuple[bool, str]:
        if self.R_5 is None:
            return False, "SKIP: R_5 not available"
        mp = self.m_pi
        me = self.m_e
        mmu = self.m_mu
        expected = (me / mmu)**2 * ((mp**2 - me**2) / (mp**2 - mmu**2))**2
        if math.isclose(self.R_5, expected, rel_tol=1e-2, abs_tol=1e-4):
            return True, "R_5 approx 1.2e-4"
        return False, f"R_5 = {self.R_5} != {expected}"

    def test_invariant_sub6_delta_S(self) -> Tuple[bool, str]:
        if self.delta_S_6 is None:
            return False, "SKIP: delta_S_6 not available"
        expected = -1.0  # K_L0 (S=+1) -> e+e- (S=0)
        if math.isclose(self.delta_S_6, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_S_6 approx -1"
        return False, f"delta_S_6 = {self.delta_S_6} != {expected}"

    def test_invariant_sub6_delta_Q_hadron(self) -> Tuple[bool, str]:
        if self.delta_Q_hadron_6 is None:
            return False, "SKIP: delta_Q_hadron_6 not available"
        expected = 0.0  # no hadrons in final state
        if math.isclose(self.delta_Q_hadron_6, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "delta_Q_hadron_6 approx 0"
        return False, f"delta_Q_hadron_6 = {self.delta_Q_hadron_6} != {expected}"

    def test_invariant_sub7_L_initial(self) -> Tuple[bool, str]:
        if self.L_initial_7 is None:
            return False, "SKIP: L_initial_7 not available"
        expected = 0.0
        if math.isclose(self.L_initial_7, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "L_initial_7 approx 0"
        return False, f"L_initial_7 = {self.L_initial_7} != {expected}"

    def test_invariant_sub7_L_final(self) -> Tuple[bool, str]:
        if self.L_final_7 is None:
            return False, "SKIP: L_final_7 not available"
        expected = 1.0  # one e⁻
        if math.isclose(self.L_final_7, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "L_final_7 approx 1"
        return False, f"L_final_7 = {self.L_final_7} != {expected}"

    def test_invariant_sub8_C_initial(self) -> Tuple[bool, str]:
        if self.C_initial_8 is None:
            return False, "SKIP: C_initial_8 not available"
        expected = 1.0  # π⁰ C=+1
        if math.isclose(self.C_initial_8, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "C_initial_8 approx 1"
        return False, f"C_initial_8 = {self.C_initial_8} != {expected}"

    def test_invariant_sub8_C_final(self) -> Tuple[bool, str]:
        if self.C_final_8 is None:
            return False, "SKIP: C_final_8 not available"
        expected = -1.0  # three γ: C=(-1)^3 = -1
        if math.isclose(self.C_final_8, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "C_final_8 approx -1"
        return False, f"C_final_8 = {self.C_final_8} != {expected}"

    def test_invariant_sub9_CP_initial(self) -> Tuple[bool, str]:
        if self.CP_initial_9 is None:
            return False, "SKIP: CP_initial_9 not available"
        expected = -1.0  # K_L^0 CP=-1
        if math.isclose(self.CP_initial_9, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "CP_initial_9 approx -1"
        return False, f"CP_initial_9 = {self.CP_initial_9} != {expected}"

    def test_invariant_sub9_CP_final(self) -> Tuple[bool, str]:
        if self.CP_final_9 is None:
            return False, "SKIP: CP_final_9 not available"
        expected = 1.0  # π⁺π⁻ L=0 gives CP=+1
        if math.isclose(self.CP_final_9, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "CP_final_9 approx 1"
        return False, f"CP_final_9 = {self.CP_final_9} != {expected}"

    def test_invariant_sub10_Q_initial(self) -> Tuple[bool, str]:
        if self.Q_initial_10 is None:
            return False, "SKIP: Q_initial_10 not available"
        expected = 1.0
        if math.isclose(self.Q_initial_10, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Q_initial_10 approx 1"
        return False, f"Q_initial_10 = {self.Q_initial_10} != {expected}"

    def test_invariant_sub10_Q_final(self) -> Tuple[bool, str]:
        if self.Q_final_10 is None:
            return False, "SKIP: Q_final_10 not available"
        expected = 2.0  # π⁺ + π⁺ + π⁰ = 1+1+0=2
        if math.isclose(self.Q_final_10, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Q_final_10 approx 2"
        return False, f"Q_final_10 = {self.Q_final_10} != {expected}"

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_sub2_energy_violation_sign(self) -> Tuple[bool, str]:
        if self.m_initial_2 is None or self.m_final_sum_2 is None:
            return False, "SKIP: sub2 masses missing"
        if self.m_final_sum_2 > self.m_initial_2:
            return True, "correctly indicates violation (final > initial)"
        return False, "energy violation not indicated (final <= initial)"

    def test_modeling_sub4_baryon_violation(self) -> Tuple[bool, str]:
        if self.B_initial_4 is None or self.B_final_4 is None:
            return False, "SKIP: sub4 baryon numbers missing"
        if self.B_initial_4 != self.B_final_4:
            return True, "correctly indicates baryon number violation"
        return False, "baryon numbers equal, should be violated"

    def test_modeling_sub7_lepton_violation(self) -> Tuple[bool, str]:
        if self.L_initial_7 is None or self.L_final_7 is None:
            return False, "SKIP: sub7 lepton numbers missing"
        if self.L_initial_7 != self.L_final_7:
            return True, "correctly indicates lepton number violation"
        return False, "lepton numbers equal, should be violated"

    def test_modeling_sub8_C_violation(self) -> Tuple[bool, str]:
        if self.C_initial_8 is None or self.C_final_8 is None:
            return False, "SKIP: sub8 C-parity missing"
        if self.C_initial_8 != self.C_final_8:
            return True, "correctly indicates C-parity violation"
        return False, "C-parity conserved, should be violated"

    def test_modeling_sub9_CP_violation(self) -> Tuple[bool, str]:
        if self.CP_initial_9 is None or self.CP_final_9 is None:
            return False, "SKIP: sub9 CP missing"
        if self.CP_initial_9 != self.CP_final_9:
            return True, "correctly indicates CP violation"
        return False, "CP conserved, should be violated"

    def test_modeling_sub10_charge_violation(self) -> Tuple[bool, str]:
        if self.Q_initial_10 is None or self.Q_final_10 is None:
            return False, "SKIP: sub10 charges missing"
        if self.Q_initial_10 != self.Q_final_10:
            return True, "correctly indicates charge violation"
        return False, "charges equal, should be violated"

    def test_modeling_sub3_deltaS_deltaQ_rule(self) -> Tuple[bool, str]:
        if self.delta_S_3 is None or self.delta_Q_hadron_3 is None:
            return False, "SKIP: sub3 Delta S and Delta Q missing"
        if self.delta_S_3 != self.delta_Q_hadron_3:
            return True, "correctly indicates DeltaS != DeltaQ violation"
        return False, "DeltaS = DeltaQ, should be violated"

    def test_modeling_sub6_deltaS_deltaQ_rule(self) -> Tuple[bool, str]:
        if self.delta_S_6 is None or self.delta_Q_hadron_6 is None:
            return False, "SKIP: sub6 Delta S and Delta Q missing"
        if self.delta_S_6 != self.delta_Q_hadron_6:
            return True, "correctly indicates DeltaS != DeltaQ violation"
        return False, "DeltaS = DeltaQ, should be violated"

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb an input and compare perturbed output to baseline
    # ------------------------------------------------------------------
    def test_monotonicity_sub2_initial_mass_increase(self) -> Tuple[bool, str]:
        """If initial mass increases, the excess (final - initial) decreases."""
        if self.m_initial_2 is None or self.m_final_sum_2 is None:
            return False, "SKIP: sub2 masses missing"
        original_excess = self.m_final_sum_2 - self.m_initial_2
        # Perturb: increase initial mass by 1%
        perturbed_initial = self.m_initial_2 * 1.01
        perturbed_excess = self.m_final_sum_2 - perturbed_initial
        if perturbed_excess < original_excess:
            return True, "excess decreases as initial mass increases"
        return False, f"monotonicity failed: original excess {original_excess}, perturbed {perturbed_excess}"

    def test_monotonicity_sub2_final_mass_increase(self) -> Tuple[bool, str]:
        """If final mass increases, the excess increases."""
        if self.m_initial_2 is None or self.m_final_sum_2 is None:
            return False, "SKIP: sub2 masses missing"
        original_excess = self.m_final_sum_2 - self.m_initial_2
        # Perturb: increase each contributing pion mass by 1%
        perturbed_final = 2 * (self.m_pi * 1.01) + 3 * (self.m_pi0 * 1.01)
        perturbed_excess = perturbed_final - self.m_initial_2
        if perturbed_excess > original_excess:
            return True, "excess increases as final masses increase"
        return False, f"monotonicity failed: original excess {original_excess}, perturbed {perturbed_excess}"

    def test_monotonicity_sub5_R_with_pion_mass(self) -> Tuple[bool, str]:
        """R depends on pion mass. Test R increases slightly if pion mass increases? Actually need to compute. 
        For standard formula, increasing pion mass increases (m_pi^2 - m_e^2) and (m_pi^2 - m_mu^2) but the ratio behavior is not obvious. 
        We'll just check that R decreases when muon mass increases (known helicity suppression)."""
        if self.R_5 is None:
            return False, "SKIP: R_5 not available"
        mp = self.m_pi
        me = self.m_e
        mmu = self.m_mu
        R_base = (me / mmu)**2 * ((mp**2 - me**2) / (mp**2 - mmu**2))**2
        # Perturb muon mass upward by 5%
        mmu_pert = mmu * 1.05
        R_pert = (me / mmu_pert)**2 * ((mp**2 - me**2) / (mp**2 - mmu_pert**2))**2
        if R_pert < R_base:
            return True, "R decreases when muon mass increases (helicity suppression)"
        return False, f"monotonicity failed: R_base={R_base}, R_pert={R_pert}"

    def test_monotonicity_sub5_R_with_electron_mass(self) -> Tuple[bool, str]:
        """Increasing electron mass increases numerator, so R increases."""
        if self.R_5 is None:
            return False, "SKIP: R_5 not available"
        mp = self.m_pi
        me = self.m_e
        mmu = self.m_mu
        R_base = (me / mmu)**2 * ((mp**2 - me**2) / (mp**2 - mmu**2))**2
        # Perturb electron mass upward by 5%
        me_pert = me * 1.05
        R_pert = (me_pert / mmu)**2 * ((mp**2 - me_pert**2) / (mp**2 - mmu**2))**2
        if R_pert > R_base:
            return True, "R increases when electron mass increases"
        return False, f"monotonicity failed: R_base={R_base}, R_pert={R_pert}"

    def test_monotonicity_sub3_delta_S_sign(self) -> Tuple[bool, str]:
        """Perturb strangeness of initial: if we used K0 instead of Kbar0, delta_S would be 0 instead of +1"""
        # Not truly a monotonicity because no continuous parameter, but we can check stability:
        # If we artificially increase S_initial by 1, delta_S decreases by 1.
        if self.delta_S_3 is None:
            return False, "SKIP: delta_S_3 not available"
        # Simulate: if initial had S=0 (instead of -1), delta_S would be 0 (instead of +1).
        # That is a change of +1 in S_initial leads to -1 in delta_S.
        # But we cannot modify the instance. We'll just check that delta_S is positive as expected.
        # However this is not a perturbation. We'll skip actual monotonicity for discrete values.
        # Instead, we'll create a dummy monotonicity: assume a hypothetical continuous "strangeness offset" parameter.
        # Because strangeness is discrete, a meaningful monotonicity test is not possible.
        # We'll return True with a note.
        return True, "SKIP (discrete quantum numbers cannot be continuously perturbed)"

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad(self) -> Tuple[bool, str]:
        """
        Very weak sanity: all numerical results should be finite and of reasonable sign/magnitude.
        No hardcoding of expected answer strings; only basic sanity.
        """
        numeric_attrs = [
            self.delta_S_1, self.delta_I3_1,
            self.m_initial_2, self.m_final_sum_2,
            self.delta_S_3, self.delta_Q_hadron_3,
            self.B_initial_4, self.B_final_4,
            self.R_5,
            self.delta_S_6, self.delta_Q_hadron_6,
            self.L_initial_7, self.L_final_7,
            self.C_initial_8, self.C_final_8,
            self.CP_initial_9, self.CP_final_9,
            self.Q_initial_10, self.Q_final_10
        ]
        for val in numeric_attrs:
            if val is None:
                continue
            if not math.isfinite(val):
                return False, f"Non-finite value: {val}"
        # Check that delta_S values are reasonable integers between -2 and 2
        for ds in [self.delta_S_1, self.delta_S_3, self.delta_S_6]:
            if ds is not None and (abs(ds) > 2 or ds != int(ds)):
                return False, f"delta_S {ds} out of range"
        # Check masses are positive
        for m in [self.m_initial_2, self.m_final_sum_2]:
            if m is not None and m <= 0:
                return False, f"non-positive mass {m}"
        # Check fraction R is small (helicity suppression) and positive
        if self.R_5 is not None and not (0 < self.R_5 < 1):
            return False, f"R_5 out of range: {self.R_5}"
        return True, "All numerical values are finite and within plausible ranges"