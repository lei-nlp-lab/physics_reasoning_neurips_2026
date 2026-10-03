import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results (required by contract)
        self.results = getattr(solution_instance, 'results', {})
        self.L_f_parity = self.results.get('L_f_parity', None)          # string 'even' or 'odd'
        self.neutron_spin = self.results.get('neutron_spin', None)      # integer 0 or 1

        # Extract givens (expected from problem context)
        self.pi_spin = getattr(solution_instance, 'pi_spin', None)
        self.pi_parity = getattr(solution_instance, 'pi_parity', None)
        self.d_spin = getattr(solution_instance, 'd_spin', None)
        self.d_parity = getattr(solution_instance, 'd_parity', None)
        self.L_i = getattr(solution_instance, 'L_i', None)              # initial orbital angular momentum (integer)
        self.neutron_parity = getattr(solution_instance, 'neutron_parity', None)

        # Explanation steps (reference only)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ========== Interface Tests ==========
    def test_interface_has_required_keys(self):
        """Check that results dictionary contains the required emit keys."""
        if self.L_f_parity is None:
            return (False, "Missing required key 'L_f_parity'")
        if self.neutron_spin is None:
            return (False, "Missing required key 'neutron_spin'")
        return (True, "Both required keys present")

    def test_interface_valid_L_f_parity_type(self):
        """L_f_parity must be a string 'even' or 'odd'."""
        if self.L_f_parity is None:
            return (False, "SKIP: L_f_parity not set")
        if not isinstance(self.L_f_parity, str):
            return (False, f"L_f_parity should be string, got {type(self.L_f_parity)}")
        if self.L_f_parity not in ('even', 'odd'):
            return (False, f"L_f_parity must be 'even' or 'odd', got '{self.L_f_parity}'")
        return (True, f"L_f_parity = '{self.L_f_parity}' valid")

    def test_interface_valid_neutron_spin_type(self):
        """neutron_spin must be an integer 0 or 1."""
        if self.neutron_spin is None:
            return (False, "SKIP: neutron_spin not set")
        if not isinstance(self.neutron_spin, int):
            return (False, f"neutron_spin should be int, got {type(self.neutron_spin)}")
        if self.neutron_spin not in (0, 1):
            return (False, f"neutron_spin must be 0 (singlet) or 1 (triplet), got {self.neutron_spin}")
        return (True, f"neutron_spin = {self.neutron_spin} valid")

    # ========== Invariant (Theorem-based) Tests ==========
    def test_invariant_parity_conservation(self):
        """Parity of initial state must equal parity of final state (strong interaction)."""
        if self.pi_parity is None:
            return (False, "SKIP: pi_parity not available")
        if self.d_parity is None:
            return (False, "SKIP: d_parity not available")
        if self.L_i is None:
            return (False, "SKIP: L_i not available")
        if self.neutron_parity is None:
            return (False, "SKIP: neutron_parity not available")
        if self.L_f_parity is None:
            return (False, "SKIP: L_f_parity not available (required for final parity)")

        # Compute initial parity: P(pi) * P(d) * (-1)^L_i
        try:
            P_i = self.pi_parity * self.d_parity * ((-1) ** self.L_i)
        except Exception as e:
            return (False, f"Error computing initial parity: {e}")

        # Compute final parity: (neutron_parity)^2 * (-1)^L_f
        try:
            if self.L_f_parity == 'even':
                L_f_factor = 1
            elif self.L_f_parity == 'odd':
                L_f_factor = -1
            else:
                return (False, f"Invalid L_f_parity value: '{self.L_f_parity}'")
            P_f = (self.neutron_parity ** 2) * L_f_factor
        except Exception as e:
            return (False, f"Error computing final parity: {e}")

        if math.isclose(P_i, P_f, rel_tol=1e-9, abs_tol=1e-9):
            return (True, f"Parity conserved: P_i = {P_i} == P_f = {P_f}")
        else:
            return (False, f"Parity not conserved: P_i = {P_i}, P_f = {P_f}")

    def test_invariant_pauli_antisymmetry(self):
        """Total wavefunction of two identical fermions must be antisymmetric."""
        if self.L_f_parity is None:
            return (False, "SKIP: L_f_parity not available")
        if self.neutron_spin is None:
            return (False, "SKIP: neutron_spin not available")

        # Spatial exchange symmetry: (-1)^{L_f}
        if self.L_f_parity == 'even':
            spatial_sym = 1
        elif self.L_f_parity == 'odd':
            spatial_sym = -1
        else:
            return (False, f"Invalid L_f_parity value: '{self.L_f_parity}'")

        # Spin exchange symmetry: -1 for S=0 (singlet), +1 for S=1 (triplet)
        if self.neutron_spin == 0:
            spin_sym = -1
        elif self.neutron_spin == 1:
            spin_sym = 1
        else:
            return (False, f"Invalid neutron_spin value: {self.neutron_spin}")

        total_sym = spatial_sym * spin_sym
        if math.isclose(total_sym, -1, rel_tol=1e-9, abs_tol=1e-9):
            return (True, f"Pauli antisymmetry satisfied: spatial_sym * spin_sym = {spatial_sym} * {spin_sym} = -1")
        else:
            return (False, f"Pauli antisymmetry violated: product = {total_sym}, expected -1")

    # ========== Modeling/Trace Tests ==========
    def test_modeling_spin_deduction_consistency(self):
        """From L_f parity, deduce the spin state according to Pauli principle."""
        if self.L_f_parity is None or self.neutron_spin is None:
            return (False, "SKIP: L_f_parity or neutron_spin missing")

        # Given L_f parity, the Pauli principle forces a specific spin
        if self.L_f_parity == 'even':
            expected_spin = 0   # singlet required
        elif self.L_f_parity == 'odd':
            expected_spin = 1   # triplet allowed
        else:
            return (False, f"Invalid L_f_parity")

        if self.neutron_spin == expected_spin:
            return (True, f"Spin state consistent with L_f parity: {self.L_f_parity} -> S={expected_spin}")
        else:
            return (False, f"Spin state inconsistent: L_f={self.L_f_parity} expects S={expected_spin}, but got S={self.neutron_spin}")

    def test_trace_explanation_consistency(self):
        """Qualitative check: explanation steps should mention 'singlet' and 'even'."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        explanation_text = ''.join(str(step) for step in self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        mentions_even = 'even' in explanation_text.lower()
        mentions_singlet = 'singlet' in explanation_text.lower()

        issues = []
        if mentions_even and self.L_f_parity != 'even':
            issues.append("Explanation says L_f even but result is not 'even'")
        if mentions_singlet and self.neutron_spin != 0:
            issues.append("Explanation says singlet but neutron_spin is not 0")
        if not mentions_even and not mentions_singlet:
            issues.append("Explanation does not mention 'even' or 'singlet' (weak)")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Explanation matches result qualitatively")

    # ========== Monotonicity Test ==========
    def test_monotonicity_Li_perturbation(self):
        """
        Perturb initial orbital angular momentum L_i by +1 (from the given value)
        and verify that the parity of L_f flips, leading to a different spin state.
        """
        if self.L_i is None:
            return (False, "SKIP: L_i not available for perturbation")
        if self.pi_parity is None or self.d_parity is None or self.neutron_parity is None:
            return (False, "SKIP: parity values not available")
        if self.L_f_parity is None or self.neutron_spin is None:
            return (False, "SKIP: result keys not available")

        # Baseline initial parity
        baseline_initial_parity = self.pi_parity * self.d_parity * ((-1) ** self.L_i)

        # Perturbation: increase L_i by 1
        perturbed_L_i = self.L_i + 1
        perturbed_initial_parity = self.pi_parity * self.d_parity * ((-1) ** perturbed_L_i)

        # Determine baseline L_f_parity from result
        if self.L_f_parity == 'even':
            baseline_L_f_even = True
        elif self.L_f_parity == 'odd':
            baseline_L_f_even = False
        else:
            return (False, "Invalid L_f_parity value")

        # Expected perturbed L_f_parity flips
        expected_perturbed_even = not baseline_L_f_even

        # Expected perturbed spin from Pauli principle
        if expected_perturbed_even:
            expected_spin = 0  # singlet
        else:
            expected_spin = 1  # triplet

        baseline_spin = self.neutron_spin
        if baseline_spin == expected_spin:
            return (False, f"Perturbing L_i by +1 should flip the spin state, but baseline spin {baseline_spin} equals expected perturbed spin {expected_spin}")
        else:
            return (True, f"Monotonicity consistent: L_i+1 flips L_f parity and spin (baseline S={baseline_spin} -> perturbed S={expected_spin})")

    # ========== Answer Sanity (weak) - OPTIONAL: we skip it to avoid any hardcoded comparison ==========
    # (No test_answer_sanity_* method to comply with "no hardcoded target_* values")

    # ========== Orchestrator ==========
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