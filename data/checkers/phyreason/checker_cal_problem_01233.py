import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with fallbacks
        self.inputs = {}
        input_keys = ['M', 'm', 'F', 'alpha', 'g']
        default_inputs = {
            'M': 2*math.sqrt(3),
            'm': math.sqrt(3),
            'F': 10*math.sqrt(3),
            'alpha': 30,  # degrees
            'g': 10
        }
        for key in input_keys:
            val = getattr(solution_instance, key, None)
            if val is None:
                val = default_inputs.get(key)
            self.inputs[key] = val

        # Convert alpha to radians
        self.alpha_rad = math.radians(self.inputs['alpha'])

        # Safe extraction of results
        self.results = {}
        result_keys = ['F_N', 'T', 'Tsintheta', 'Tcostheta', 'F_f', 'theta', 'mu']
        for key in result_keys:
            val = getattr(solution_instance, key, None)
            if val is None and hasattr(solution_instance, 'results'):
                val = solution_instance.results.get(key, None)
            self.results[key] = val

        # Convert theta to radians if available
        if self.results['theta'] is not None:
            theta_val = self.results['theta']
            if isinstance(theta_val, (int, float)) and abs(theta_val) > 2*math.pi:
                self.theta_rad = math.radians(theta_val)
            else:
                self.theta_rad = float(theta_val)  # assume radians
        else:
            self.theta_rad = None

    def test_invariants(self):
        logs = []
        success = True

        M = self.inputs['M']
        m = self.inputs['m']
        F = self.inputs['F']
        alpha = self.alpha_rad
        g = self.inputs['g']

        F_N = self.results['F_N']
        T = self.results['T']
        T_sin = self.results['Tsintheta']
        T_cos = self.results['Tcostheta']
        F_f = self.results['F_f']
        mu = self.results['mu']

        # 1. Ball horizontal equilibrium: F*cos(alpha) = T*cos(theta)
        if T_cos is not None:
            lhs = F * math.cos(alpha)
            rhs = T_cos
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Ball horizontal equilibrium: {lhs} vs {rhs}")
                success = False
            else:
                logs.append("PASS: Ball horizontal equilibrium")
        else:
            logs.append("SKIP: T*cos(theta) missing")

        # 2. Ball vertical equilibrium: T*sin(theta) + F*sin(alpha) = m*g
        if T_sin is not None:
            lhs = T_sin + F * math.sin(alpha)
            rhs = m * g
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Ball vertical equilibrium: {lhs} vs {rhs}")
                success = False
            else:
                logs.append("PASS: Ball vertical equilibrium")
        else:
            logs.append("SKIP: T*sin(theta) missing")

        # 3. System vertical equilibrium: F_N + F*sin(alpha) = (m+M)*g
        if F_N is not None:
            lhs = F_N + F * math.sin(alpha)
            rhs = (m + M) * g
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: System vertical equilibrium: {lhs} vs {rhs}")
                success = False
            else:
                logs.append("PASS: System vertical equilibrium")
        else:
            logs.append("SKIP: F_N missing")

        # 4. System horizontal equilibrium: F_f = F*cos(alpha)
        if F_f is not None:
            lhs = F_f
            rhs = F * math.cos(alpha)
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: System horizontal equilibrium: {lhs} vs {rhs}")
                success = False
            else:
                logs.append("PASS: System horizontal equilibrium")
        else:
            logs.append("SKIP: F_f missing")

        # 5. Friction definition: F_f = μ * F_N
        if all(v is not None for v in [F_f, F_N, mu]):
            lhs = F_f
            rhs = mu * F_N
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction definition: {lhs} vs {rhs}")
                success = False
            else:
                logs.append("PASS: Friction definition")
        else:
            logs.append("SKIP: F_f, F_N, or μ missing")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        M = self.inputs['M']
        m = self.inputs['m']
        F = self.inputs['F']
        alpha = self.alpha_rad
        g = self.inputs['g']

        F_N = self.results['F_N']
        T = self.results['T']
        T_sin = self.results['Tsintheta']
        T_cos = self.results['Tcostheta']
        F_f = self.results['F_f']
        mu = self.results['mu']

        # Consistency between T, theta, and their components
        if T is not None and self.theta_rad is not None:
            exp_sin = T * math.sin(self.theta_rad)
            exp_cos = T * math.cos(self.theta_rad)
            if T_sin is not None and not math.isclose(T_sin, exp_sin, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T*sin(theta) inconsistent: {T_sin} vs {exp_sin}")
                success = False
            if T_cos is not None and not math.isclose(T_cos, exp_cos, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T*cos(theta) inconsistent: {T_cos} vs {exp_cos}")
                success = False
            if T_sin is not None and T_cos is not None:
                logs.append("PASS: T and theta components consistent")
        else:
            logs.append("SKIP: T or theta missing for component check")

        # μ = F_f / F_N
        if all(v is not None for v in [F_f, F_N, mu]):
            computed_mu = F_f / F_N
            if not math.isclose(mu, computed_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: μ mismatch: {mu} vs {computed_mu}")
                success = False
            else:
                logs.append("PASS: μ consistent with F_f and F_N")
        else:
            logs.append("SKIP: Missing values for μ check")

        # Compute θ and T from ball equilibrium - CORRECTED VERSION
        # From equations: F*cosα = T*cosθ, and T*sinθ + F*sinα = m*g
        # Let A = F*cosα, B = m*g - F*sinα
        # Then T*cosθ = A, T*sinθ = B
        # So θ = atan2(B, A), T = sqrt(A^2 + B^2)
        A = F * math.cos(alpha)
        B = m * g - F * math.sin(alpha)
        if abs(A) > 1e-10:
            theta_computed = math.atan2(B, A)
            T_computed = math.sqrt(A*A + B*B)
            if self.theta_rad is not None and not math.isclose(self.theta_rad, theta_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: θ inconsistent: {self.theta_rad} vs {theta_computed}")
                success = False
            if T is not None and not math.isclose(T, T_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T inconsistent: {T} vs {T_computed}")
                success = False
            if self.theta_rad is not None or T is not None:
                logs.append("PASS: θ and T consistent with ball equilibrium")
        else:
            logs.append("WARNING: A is zero, skipping θ and T computation")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        M = self.inputs['M']
        m = self.inputs['m']
        F = self.inputs['F']
        alpha = self.alpha_rad
        g = self.inputs['g']

        mu_orig = self.results['mu']
        if mu_orig is None:
            logs.append("SKIP: μ not available")
            return False, logs

        # Derived formula: μ = F*cos(α) / [(m+M)*g - F*sin(α)]
        def compute_mu(mass):
            numerator = F * math.cos(alpha)
            denominator = (mass + M) * g - F * math.sin(alpha)
            if abs(denominator) < 1e-10:
                return float('inf')
            return numerator / denominator

        # Increase m by 10%
        m_inc = m * 1.1
        mu_inc = compute_mu(m_inc)
        if not (mu_inc < mu_orig):  # Use strict inequality with tolerance
            logs.append(f"FAIL: Increasing m should decrease μ. μ_orig={mu_orig:.6f}, μ_inc={mu_inc:.6f}")
            success = False
        else:
            logs.append(f"PASS: Increasing m decreases μ: {mu_orig:.6f} → {mu_inc:.6f}")

        # Decrease m by 10%
        m_dec = m * 0.9
        mu_dec = compute_mu(m_dec)
        if not (mu_dec > mu_orig):  # Use strict inequality with tolerance
            logs.append(f"FAIL: Decreasing m should increase μ. μ_orig={mu_orig:.6f}, μ_dec={mu_dec:.6f}")
            success = False
        else:
            logs.append(f"PASS: Decreasing m increases μ: {mu_orig:.6f} → {mu_dec:.6f}")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test in tests:
            all_logs.append(f"\n=== {name} ===")
            try:
                success, logs = test()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                overall_success = False
                all_logs.append(f"ERROR: {e}")
                all_logs.append(traceback.format_exc())

        return overall_success, all_logs