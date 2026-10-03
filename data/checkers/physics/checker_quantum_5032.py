import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a helium atom solution for quantum numbers, Pauli principle, 
    and perturbation formulas with direct and exchange integrals.
    """

    def __init__(self, solution_instance):
        # Safely extract results and explanation steps
        results = getattr(solution_instance, 'results', {})
        self.ground = results.get('ground', None)
        self.first_excited = results.get('first_excited', None)
        self.second_excited = results.get('second_excited', None)
        self.pauli_effect = results.get('pauli_effect', None)
        self.delta_E_1s2 = results.get('Delta_E_1s2', None)
        self.delta_E_singlet = results.get('Delta_E_1snl_singlet', None)
        self.delta_E_triplet = results.get('Delta_E_1snl_triplet', None)

        # Explanation steps (list of strings or dicts) – may be used for trace checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ------------------------------------------------------------------
    # Interface tests – ensure required keys exist and have correct types
    # ------------------------------------------------------------------

    def test_interface_ground_state(self):
        """Check that ground state is a tuple of length 3."""
        if self.ground is None:
            return (False, "SKIP: ground state not provided")
        if not isinstance(self.ground, (tuple, list)) or len(self.ground) != 3:
            return (False, f"ground state should be a 3-element tuple, got {self.ground}")
        return (True, "ground state is a 3-element tuple")

    def test_interface_first_excited(self):
        if self.first_excited is None:
            return (False, "SKIP: first_excited not provided")
        if not isinstance(self.first_excited, (tuple, list)) or len(self.first_excited) != 3:
            return (False, f"first_excited should be a 3-element tuple, got {self.first_excited}")
        return (True, "first_excited is a 3-element tuple")

    def test_interface_second_excited(self):
        if self.second_excited is None:
            return (False, "SKIP: second_excited not provided")
        if not isinstance(self.second_excited, (tuple, list)) or len(self.second_excited) != 3:
            return (False, f"second_excited should be a 3-element tuple, got {self.second_excited}")
        return (True, "second_excited is a 3-element tuple")

    def test_interface_pauli_effect(self):
        if self.pauli_effect is None:
            return (False, "SKIP: pauli_effect not provided")
        if not isinstance(self.pauli_effect, str):
            return (False, f"pauli_effect should be a string, got {type(self.pauli_effect)}")
        return (True, "pauli_effect is a string")

    def test_interface_delta_E_expressions(self):
        """Check that all three perturbation expressions are present and are strings (or sympy objects)."""
        if self.delta_E_1s2 is None:
            return (False, "SKIP: Delta_E_1s2 not provided")
        if self.delta_E_singlet is None:
            return (False, "SKIP: Delta_E_1snl_singlet not provided")
        if self.delta_E_triplet is None:
            return (False, "SKIP: Delta_E_1snl_triplet not provided")
        # They can be strings or sympy expressions; just check they exist
        return (True, "All three Delta_E expressions are present")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------

    def test_invariant_parity_ground(self):
        """Parity (even/odd) must match (-1)^(l1+l2) for ground state (1s2: l1=0,l2=0 => even)."""
        if self.ground is None:
            return (False, "SKIP: ground state missing")
        J, S, P = self.ground
        # parity check only
        expected_parity = 'even'
        if P != expected_parity:
            return (False, f"Ground parity should be {expected_parity}, got {P}")
        return (True, "Ground parity is even (correct)")

    def test_invariant_parity_first_excited(self):
        """First excited (1s2s): l1=0,l2=0 => even."""
        if self.first_excited is None:
            return (False, "SKIP: first_excited missing")
        J, S, P = self.first_excited
        expected_parity = 'even'
        if P != expected_parity:
            return (False, f"First excited parity should be {expected_parity}, got {P}")
        return (True, "First excited parity is even")

    def test_invariant_parity_second_excited(self):
        """Second excited (1s2p): l1=0,l2=1 => odd."""
        if self.second_excited is None:
            return (False, "SKIP: second_excited missing")
        J, S, P = self.second_excited
        expected_parity = 'odd'
        if P != expected_parity:
            return (False, f"Second excited parity should be {expected_parity}, got {P}")
        return (True, "Second excited parity is odd")

    def test_invariant_spin_excited_states(self):
        """First and second excited should be triplets (S=1) because they are the lowest energy states of those configurations."""
        if self.first_excited is None or self.second_excited is None:
            return (False, "SKIP: excited states missing")
        J1, S1, P1 = self.first_excited
        J2, S2, P2 = self.second_excited
        if S1 != 1:
            return (False, f"First excited S should be 1 (triplet), got {S1}")
        if S2 != 1:
            return (False, f"Second excited S should be 1 (triplet), got {S2}")
        return (True, "Both excited states are triplets (S=1)")

    def test_invariant_J_ground(self):
        """Ground state: L=0, S=0 => J=0."""
        if self.ground is None:
            return (False, "SKIP: ground missing")
        J, S, _ = self.ground
        if J != 0 and S == 0:
            return (False, f"Ground J should be 0, got {J}")
        return (True, "Ground J=0")

    def test_invariant_J_first_excited(self):
        """First excited (1s2s triplet): L=0, S=1 => J=1."""
        if self.first_excited is None:
            return (False, "SKIP: first_excited missing")
        J, S, _ = self.first_excited
        if S == 1 and J != 1:
            return (False, f"For 1s2s triplet, J should be 1, got {J}")
        return (True, "First excited J=1 (consistent with triple L=0)")

    def test_invariant_J_second_excited(self):
        """Second excited (1s2p triplet): L=1, S=1 => lowest J = |L-S| = 0."""
        if self.second_excited is None:
            return (False, "SKIP: second_excited missing")
        J, S, _ = self.second_excited
        if S == 1 and J != 0:
            return (False, f"For 1s2p triplet lowest J should be 0, got {J}")
        return (True, "Second excited J=0 (lowest from L=1, S=1)")

    def test_invariant_pauli_principle_is_mentioned(self):
        """The Pauli principle explanation should mention that triplet is lower energy."""
        if self.pauli_effect is None:
            return (False, "SKIP: pauli_effect missing")
        if "triplet" not in self.pauli_effect.lower():
            return (False, "pauli_effect does not mention 'triplet' (lower energy)")
        return (True, "pauli_effect mentions triplet")

    # ------------------------------------------------------------------
    # Modeling / Trace consistency tests
    # ------------------------------------------------------------------

    def test_trace_delta_E_1s2_contains_direct(self):
        """Delta_E_1s2 should be a direct integral expression (no exchange)."""
        if self.delta_E_1s2 is None:
            return (False, "SKIP: Delta_E_1s2 missing")
        expr_str = str(self.delta_E_1s2)
        if "1s1s" not in expr_str and "direct" not in expr_str.lower():
            return (False, "Delta_E_1s2 should mention 1s1s configuration or direct integral")
        return (True, "Delta_E_1s2 appears to be a direct integral")

    def test_trace_delta_E_singlet_has_plus_k(self):
        """Singlet correction should be J + K (direct + exchange)."""
        if self.delta_E_singlet is None:
            return (False, "SKIP: Delta_E_singlet missing")
        expr_str = str(self.delta_E_singlet)
        if '+' in expr_str or 'plus' in expr_str.lower():
            return (True, "Singlet expression contains plus (J+K)")
        try:
            expr = sp.sympify(expr_str)
            if isinstance(expr, sp.Add):
                return (True, "Singlet expression is a sum (J+K)")
        except:
            pass
        return (False, "Singlet expression does not appear to be J+K")

    def test_trace_delta_E_triplet_has_minus_k(self):
        """Triplet correction should be J - K (direct - exchange)."""
        if self.delta_E_triplet is None:
            return (False, "SKIP: Delta_E_triplet missing")
        expr_str = str(self.delta_E_triplet)
        if '-' in expr_str or 'minus' in expr_str.lower():
            return (True, "Triplet expression contains minus (J-K)")
        try:
            expr = sp.sympify(expr_str)
            if isinstance(expr, sp.Add) and len(expr.args) == 2:
                return (True, "Triplet expression could be J-K")
        except:
            pass
        return (False, "Triplet expression does not appear to be J-K")

    def test_trace_explanation_steps_contain_antisymmetry(self):
        """Explanation steps should mention antisymmetric spatial wavefunction."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = " ".join(str(s) for s in self.explanation_steps)
        if "antisymmetric" not in text.lower():
            return (False, "Explanation steps do not mention antisymmetric spatial wavefunction")
        return (True, "Explanation steps mention antisymmetry")

    # ------------------------------------------------------------------
    # Monotonicity test – explicit perturbation of exchange integral K
    # ------------------------------------------------------------------

    def test_monotonicity_exchange_effect(self):
        """
        Perturb the exchange integral K (increase it) and verify that the
        singlet energy correction increases while the triplet energy correction decreases.
        This tests that the coefficient of K is +1 in singlet and -1 in triplet.
        """
        if self.delta_E_singlet is None or self.delta_E_triplet is None:
            return (False, "SKIP: Delta_E_singlet or Delta_E_triplet missing")

        # Define symbolic variables for J and K
        J_sym, K_sym = sp.symbols('J K')
        # Convert expressions to sympy if possible
        try:
            expr_singlet = sp.sympify(str(self.delta_E_singlet))
            expr_triplet = sp.sympify(str(self.delta_E_triplet))
        except:
            return (False, "Unable to parse expressions as symbolic; cannot perform monotonicity test")

        # Substitute J=1, K=0.1 (arbitrary positive values)
        subs_initial = {J_sym: 1, K_sym: 0.1}
        subs_perturbed = {J_sym: 1, K_sym: 0.2}  # double K

        val_singlet_initial = float(expr_singlet.subs(subs_initial).evalf())
        val_singlet_perturbed = float(expr_singlet.subs(subs_perturbed).evalf())
        val_triplet_initial = float(expr_triplet.subs(subs_initial).evalf())
        val_triplet_perturbed = float(expr_triplet.subs(subs_perturbed).evalf())

        singlet_increases = val_singlet_perturbed > val_singlet_initial
        triplet_decreases = val_triplet_perturbed < val_triplet_initial

        if singlet_increases and triplet_decreases:
            return (True, "Singlet increases and triplet decreases when K increases (correct J+K vs J-K behavior)")
        else:
            msg = f"Singlet initial={val_singlet_initial}, perturbed={val_singlet_perturbed}, increases={singlet_increases}; "
            msg += f"Triplet initial={val_triplet_initial}, perturbed={val_triplet_perturbed}, decreases={triplet_decreases}"
            return (False, msg)

    # ------------------------------------------------------------------
    # Weak answer sanity (only one test, no numeric derivation)
    # ------------------------------------------------------------------

    def test_answer_sanity_quantum_numbers(self):
        """
        Verify that all J, S, P are within expected ranges.
        J in {0,1,2}, S in {0,1}, P in {'even','odd'}.
        """
        states = [("ground", self.ground), ("first_excited", self.first_excited), ("second_excited", self.second_excited)]
        for name, state in states:
            if state is None:
                continue
            J, S, P = state
            if J not in (0, 1, 2):
                return (False, f"{name}: J={J} out of expected range [0,1,2]")
            if S not in (0, 1):
                return (False, f"{name}: S={S} out of expected {0,1}")
            if P not in ('even', 'odd'):
                return (False, f"{name}: P={P} should be 'even' or 'odd'")
        return (True, "All quantum numbers in plausible range")