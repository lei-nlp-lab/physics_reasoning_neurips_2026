import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.Q_H = self.results.get('Q_H', None)
        self.S_H = self.results.get('S_H', None)
        self.B_H = self.results.get('B_H', None)
        self.n_quarks = self.results.get('n_quarks', None)
        self.p_beam = self.results.get('p_beam', None)
        self.strong_decays = self.results.get('strong_decays', None)
        self.weak_decays = self.results.get('weak_decays', None)

        # Physics constants (MeV, c=1)
        self.m_p = 938.272
        self.m_K = 493.677
        self.m_H = 2150.0
        self.m_n = 939.565
        self.m_Lambda = 1115.683
        self.m_Sigma0 = 1192.642
        self.m_Sigma_minus = 1197.449

    # ---- Interface checks ----
    def test_interface_q_h(self):
        if self.Q_H is None:
            return (False, "SKIP: Q_H not provided")
        if not isinstance(self.Q_H, (int, float)):
            return (False, "Q_H must be numeric")
        return (True, f"Q_H = {self.Q_H}")

    def test_interface_s_h(self):
        if self.S_H is None:
            return (False, "SKIP: S_H not provided")
        if not isinstance(self.S_H, (int, float)):
            return (False, "S_H must be numeric")
        return (True, f"S_H = {self.S_H}")

    def test_interface_b_h(self):
        if self.B_H is None:
            return (False, "SKIP: B_H not provided")
        if not isinstance(self.B_H, (int, float)):
            return (False, "B_H must be numeric")
        return (True, f"B_H = {self.B_H}")

    def test_interface_n_quarks(self):
        if self.n_quarks is None:
            return (False, "SKIP: n_quarks not provided")
        if not isinstance(self.n_quarks, int):
            return (False, "n_quarks must be integer")
        return (True, f"n_quarks = {self.n_quarks}")

    def test_interface_p_beam(self):
        if self.p_beam is None:
            return (False, "SKIP: p_beam not provided")
        if not isinstance(self.p_beam, (int, float)):
            return (False, "p_beam must be numeric")
        if self.p_beam <= 0:
            return (False, "p_beam must be positive")
        if not math.isfinite(self.p_beam):
            return (False, "p_beam must be finite")
        return (True, f"p_beam = {self.p_beam}")

    def test_interface_strong_decays(self):
        if self.strong_decays is None:
            return (False, "SKIP: strong_decays not provided")
        if not isinstance(self.strong_decays, list):
            return (False, "strong_decays must be a list")
        if len(self.strong_decays) == 0:
            return (False, "strong_decays list should not be empty")
        return (True, f"strong_decays: {self.strong_decays}")

    def test_interface_weak_decays(self):
        if self.weak_decays is None:
            return (False, "SKIP: weak_decays not provided")
        if not isinstance(self.weak_decays, list):
            return (False, "weak_decays must be a list")
        if len(self.weak_decays) == 0:
            return (False, "weak_decays list should not be empty")
        return (True, f"weak_decays: {self.weak_decays}")

    # ---- Invariant checks (physical laws) ----
    def test_invariant_charge_conservation(self):
        if self.Q_H is None:
            return (False, "SKIP: Q_H not available")
        expected_Q_H = 0  # from 2 = Q_H + 2 => Q_H = 0
        if math.isclose(self.Q_H, expected_Q_H, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_H = {self.Q_H} (expected ~{expected_Q_H})")
        else:
            return (False, f"Q_H = {self.Q_H} violates charge conservation (expected {expected_Q_H})")

    def test_invariant_strangeness_conservation(self):
        if self.S_H is None:
            return (False, "SKIP: S_H not available")
        expected_S_H = -2  # 0 = S_H + 2 => S_H = -2
        if math.isclose(self.S_H, expected_S_H, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S_H = {self.S_H} (expected ~{expected_S_H})")
        else:
            return (False, f"S_H = {self.S_H} violates strangeness conservation (expected {expected_S_H})")

    def test_invariant_baryon_number_conservation(self):
        if self.B_H is None:
            return (False, "SKIP: B_H not available")
        expected_B_H = 2  # 2 = B_H + 0 => B_H = 2
        if math.isclose(self.B_H, expected_B_H, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B_H = {self.B_H} (expected ~{expected_B_H})")
        else:
            return (False, f"B_H = {self.B_H} violates baryon number conservation (expected {expected_B_H})")

    def test_invariant_threshold_condition(self):
        if self.p_beam is None or self.m_p is None or self.m_K is None or self.m_H is None:
            return (False, "SKIP: missing parameters for threshold check")
        # Compute invariant s from given p_beam
        try:
            E_beam = math.sqrt(self.p_beam**2 + self.m_p**2)
            s_from_p = (E_beam + self.m_p)**2 - self.p_beam**2
            s_expected = (self.m_H + 2 * self.m_K)**2
            if math.isclose(s_from_p, s_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"s = {s_from_p:.1f} (invariant matches threshold)")
            else:
                return (False, f"s = {s_from_p:.1f}, expected ~{s_expected:.1f} (threshold condition violated)")
        except Exception as e:
            return (False, f"Threshold check error: {e}")

    # ---- Modeling checks ----
    def test_modeling_quark_count(self):
        if self.n_quarks is None or self.B_H is None:
            return (False, "SKIP: n_quarks or B_H not available")
        expected = 3 * self.B_H
        if self.n_quarks == expected:
            return (True, f"n_quarks = {self.n_quarks} (3 * B_H = {expected})")
        else:
            return (False, f"n_quarks = {self.n_quarks}, should be {expected}")

    def test_modeling_strong_decay_masses(self):
        if self.strong_decays is None or self.m_H is None:
            return (False, "SKIP: strong_decays or m_H not available")
        # Define particle masses in MeV (using known values)
        mass_map = {
            'Lambda': self.m_Lambda,
            'n': self.m_n,
            'Sigma0': self.m_Sigma0,
            'Sigma-': self.m_Sigma_minus,
            'p': self.m_p
        }
        # Expected allowed channels: ['Lambda n', 'Sigma0 n', 'Sigma- p']
        expected_channels = ['Lambda n', 'Sigma0 n', 'Sigma- p']
        # Verify that each provided decay consists of two particles whose masses sum < m_H
        for channel in self.strong_decays:
            parts = channel.split()
            if len(parts) != 2:
                return (False, f"Decay channel '{channel}' should have exactly two particles")
            m1 = mass_map.get(parts[0])
            m2 = mass_map.get(parts[1])
            if m1 is None or m2 is None:
                return (False, f"Unknown particle in '{channel}'")
            total_mass = m1 + m2
            if not math.isclose(total_mass, self.m_H, rel_tol=1e-2, abs_tol=1e-2) and total_mass >= self.m_H:
                return (False, f"Decay '{channel}' mass sum {total_mass:.1f} >= {self.m_H} (exceeds m_H)")
        # Also check that the list matches expected channels (order not crucial)
        if set(self.strong_decays) != set(expected_channels):
            return (False, f"Strong decay list mismatch. Expected: {expected_channels}, Got: {self.strong_decays}")
        return (True, "All strong decay masses and channels consistent")

    def test_modeling_strong_decay_quantum_numbers(self):
        if self.strong_decays is None:
            return (False, "SKIP: strong_decays not available")
        # Each channel must conserve Q, S, B: H (Q=0,S=-2,B=2) -> products.
        # For each expected channel, check:
        # 'Lambda n': Q: 0+0=0, S: -1-1=-2, B: 1+1=2
        # 'Sigma0 n': Q:0+0=0, S: -1-1=-2, B:1+1=2
        # 'Sigma- p': Q: -1+1=0, S: -1-1=-2, B:1+1=2
        channel_qns = {
            'Lambda n': {'Q':0,'S':-2,'B':2},
            'Sigma0 n': {'Q':0,'S':-2,'B':2},
            'Sigma- p': {'Q':0,'S':-2,'B':2}
        }
        for channel in self.strong_decays:
            if channel not in channel_qns:
                # Allow other channels if they also conserve? But expected set.
                return (False, f"Unexpected strong decay channel: {channel}")
            # Could add detailed Q,S,B check; for now just ensure it's in allowed set
        return (True, "Strong decay quantum numbers conserved")

    def test_modeling_weak_decay_semileptonic(self):
        if self.weak_decays is None:
            return (False, "SKIP: weak_decays not available")
        # Expected weak decays: H->Lambda p e nu, H->Sigma0 p e nu
        # Check that each decay contains a lepton and neutrino, and baryon number B=2.
        expected = ['H -> Lambda p e nu', 'H -> Sigma0 p e nu']
        if set(self.weak_decays) != set(expected):
            # Allow minor formatting differences; parse
            cleaned = []
            for d in self.weak_decays:
                # Remove spaces, check for keywords
                d_clean = d.replace(' ', '')
                if 'Lambda' in d_clean and 'e' in d_clean and 'nu' in d_clean:
                    cleaned.append('H -> Lambda p e nu')
                elif 'Sigma0' in d_clean and 'e' in d_clean and 'nu' in d_clean:
                    cleaned.append('H -> Sigma0 p e nu')
            if set(cleaned) != set(expected):
                return (False, f"Weak decays do not match expected: {expected}, got {self.weak_decays}")
        return (True, "Weak decays include semileptonic modes consistent with expectation")

    # ---- Monotonicity checks ----
    def test_monotonicity_p_beam_increases_with_m_H(self):
        if self.p_beam is None or self.m_p is None or self.m_K is None:
            return (False, "SKIP: missing parameters for monotonicity")
        # Perturb m_H by +1% and recompute threshold p_beam
        try:
            original_m_H = self.m_H
            # Use the threshold formula: s = (m_H+2m_K)^2
            # E0 = (s - 2m_p^2)/(2m_p)
            # p = sqrt(E0^2 - m_p^2)
            def compute_p(m_H_val):
                s = (m_H_val + 2 * self.m_K)**2
                E0 = (s - 2 * self.m_p**2) / (2 * self.m_p)
                return math.sqrt(E0**2 - self.m_p**2)
            p_original = compute_p(original_m_H)
            # check that given p_beam matches original computed (within tolerance)
            if not math.isclose(self.p_beam, p_original, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Given p_beam {self.p_beam} does not match computed {p_original:.3f} from threshold formula")
            # Now perturb m_H upward by 1%
            p_perturbed = compute_p(original_m_H * 1.01)
            if p_perturbed <= p_original:
                return (False, "p_beam should increase when m_H increases")
            return (True, "p_beam monotonic increasing with m_H verified")
        except Exception as e:
            return (False, f"Monotonicity check error: {e}")

    # ---- Weak answer sanity (only one) ----
    def test_answer_sanity_weak(self):
        # Only check that final answers are of correct type and not obviously wrong
        checks = []
        if self.Q_H is not None:
            if isinstance(self.Q_H, (int, float)) and math.isfinite(self.Q_H):
                checks.append("Q_H valid")
            else:
                return (False, "Q_H not valid number")
        if self.S_H is not None:
            if isinstance(self.S_H, (int, float)) and math.isfinite(self.S_H):
                checks.append("S_H valid")
            else:
                return (False, "S_H not valid number")
        if self.B_H is not None:
            if isinstance(self.B_H, (int, float)) and math.isfinite(self.B_H):
                checks.append("B_H valid")
            else:
                return (False, "B_H not valid number")
        if self.n_quarks is not None:
            if isinstance(self.n_quarks, int) and self.n_quarks > 0:
                checks.append("n_quarks valid")
            else:
                return (False, "n_quarks not positive integer")
        if self.p_beam is not None:
            if isinstance(self.p_beam, (int, float)) and self.p_beam > 0 and math.isfinite(self.p_beam):
                checks.append("p_beam valid")
            else:
                return (False, "p_beam not valid positive number")
        if self.strong_decays is not None:
            if isinstance(self.strong_decays, list) and len(self.strong_decays) > 0:
                checks.append("strong_decays list nonempty")
            else:
                return (False, "strong_decays list invalid")
        if self.weak_decays is not None:
            if isinstance(self.weak_decays, list) and len(self.weak_decays) > 0:
                checks.append("weak_decays list nonempty")
            else:
                return (False, "weak_decays list invalid")
        return (True, "; ".join(checks))

    # ---- Orchestrator ----
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