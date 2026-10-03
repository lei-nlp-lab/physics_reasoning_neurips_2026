import math
import sympy as sp
from sympy.vector import cross

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Physical constants (with safe defaults)
        self.g = 9.8  # gravitational acceleration
        self.R0 = 0.1  # rod resistance (from problem statement)
        
        # Problem parameters
        self.r = 0.2  # m
        self.omega = 600  # rad/s
        self.B1 = 1.0  # T
        self.B2 = 1.0  # T
        self.l1 = 0.25  # m
        self.l2 = 0.068  # m
        self.l3 = 0.08  # m
        self.m = 0.01  # kg
        self.C = 0.09  # F
        self.R = 0.1  # Ω
        self.h = 0.2  # m (center of mass rise)
        
        # Extract results with safe fallbacks
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.Q = getattr(solution_instance, 'Q', self.results.get('Q', 0.0))
        self.DeltaQ = getattr(solution_instance, 'DeltaQ', self.results.get('DeltaQ', 0.0))
        self.v1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0))
        self.Deltax = getattr(solution_instance, 'Deltax', self.results.get('Deltax', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        
        # For symbolic computations
        self.t, self.m_sym, self.g_sym = sp.symbols('t m g', real=True, positive=True)
        self.v_sym, self.B_sym, self.l_sym = sp.symbols('v B l', real=True, positive=True)
        self.Q_sym, self.deltaQ_sym = sp.symbols('Q DeltaQ', real=True)
        self.R_sym = sp.symbols('R', real=True, positive=True)
        self.h_sym = sp.symbols('h', real=True, positive=True)

    def test_invariants(self):
        logs = []
        
        # 1. Energy conservation: kinetic energy before rise = potential energy at max height
        # Step 7: 1/2 * (2m) * v2^2 = 2m * g * h
        ke_before = 0.5 * (2*self.m) * self.v2**2
        pe_max = (2*self.m) * self.g * self.h
        
        if not math.isclose(ke_before, pe_max, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy conservation failed: KE_before={ke_before:.4f}, PE_max={pe_max:.4f}")
        
        # 2. Momentum conservation during inelastic collision
        # Step 6: m * v1 = (m + m) * v2
        p_before = self.m * self.v1
        p_after = (2*self.m) * self.v2
        
        if not math.isclose(p_before, p_after, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Momentum conservation failed: p_before={p_before:.4f}, p_after={p_after:.4f}")
        
        # 3. Faraday's law consistency
        # Step 2: E = 1/2 * B1 * omega * r^2
        E_calc = 0.5 * self.B1 * self.omega * self.r**2
        
        if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Faraday's law mismatch: E_result={self.E}, E_calc={E_calc:.4f}")
        
        # 4. Capacitance formula
        # Step 3: Q = C * U, where U = E/2 (half the induced EMF)
        U = self.E / 2
        Q_calc = self.C * U
        
        if not math.isclose(self.Q, Q_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Capacitance formula mismatch: Q_result={self.Q}, Q_calc={Q_calc:.4f}")
        
        # 5. Impulse-momentum theorem for rod ejection
        # Step 5: B2 * l1 * DeltaQ = m * v1
        impulse_calc = self.B2 * self.l1 * self.DeltaQ
        momentum_change = self.m * self.v1
        
        if not math.isclose(impulse_calc, momentum_change, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Impulse-momentum theorem mismatch: impulse={impulse_calc:.4f}, Δp={momentum_change:.4f}")
        
        # 6. Deceleration in magnetic field (step 9)
        # (B2^2 * l1^2 * Δx) / (2R) = 2m * v2
        lhs = (self.B2**2 * self.l1**2 * self.Deltax) / (2 * self.R)
        rhs = 2 * self.m * self.v2
        
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Deceleration equation mismatch: LHS={lhs:.4f}, RHS={rhs:.4f}")
        
        return logs

    def test_trace_consistency(self):
        logs = []
        
        # 1. Check v2 from energy conservation vs from momentum conservation
        v2_energy = math.sqrt(2 * self.g * self.h)  # from step 7: v2 = sqrt(2gh)
        v2_momentum = self.v1 / 2  # from step 6: v2 = v1/2
        
        if not math.isclose(v2_energy, v2_momentum, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v2 inconsistency: from energy={v2_energy:.4f}, from momentum={v2_momentum:.4f}")
        
        # 2. Check v1 from impulse-momentum theorem
        v1_calc = (self.B2 * self.l1 * self.DeltaQ) / self.m
        
        if not math.isclose(self.v1, v1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v1 inconsistency: result={self.v1}, calculated={v1_calc:.4f}")
        
        # 3. Check Δx from deceleration equation
        Deltax_calc = (2 * self.R * 2 * self.m * self.v2) / (self.B2**2 * self.l1**2)
        
        if not math.isclose(self.Deltax, Deltax_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Δx inconsistency: result={self.Deltax}, calculated={Deltax_calc:.4f}")
        
        # 4. Check final x
        x_calc = self.Deltax + (self.l3 - self.l2)
        
        if not math.isclose(self.x, x_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"x inconsistency: result={self.x}, calculated={x_calc:.4f}")
        
        # 5. Check ΔQ from step 8 formula
        DeltaQ_calc = (2 * self.m * math.sqrt(2 * self.g * self.h)) / (self.B2 * self.l1)
        
        if not math.isclose(self.DeltaQ, DeltaQ_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ΔQ inconsistency: result={self.DeltaQ}, calculated={DeltaQ_calc:.4f}")
        
        return logs

    def test_monotonicity(self):
        logs = []
        
        # Perturb B2 and check effect on x
        original_B2 = self.B2
        perturbed_B2 = original_B2 * 1.05  # +5% increase
        
        try:
            # Recalculate chain with perturbed B2
            # Step 8: ΔQ' = (2m * sqrt(2gh)) / (B2' * l1)
            DeltaQ_prime = (2 * self.m * math.sqrt(2 * self.g * self.h)) / (perturbed_B2 * self.l1)
            
            # Step 5: v1' = (B2' * l1 * ΔQ') / m
            v1_prime = (perturbed_B2 * self.l1 * DeltaQ_prime) / self.m
            
            # Step 6: v2' = v1' / 2
            v2_prime = v1_prime / 2
            
            # Step 9: Δx' = (2R * 2m * v2') / (B2'^2 * l1^2)
            Deltax_prime = (2 * self.R * 2 * self.m * v2_prime) / (perturbed_B2**2 * self.l1**2)
            
            # Step 11: x' = Δx' + (l3 - l2)
            x_prime = Deltax_prime + (self.l3 - self.l2)
            
            # Check monotonic relationship: x should decrease when B2 increases
            # Because Δx ∝ 1/B2^2 from step 9
            if x_prime > self.x:
                logs.append(f"Monotonicity check failed: x increased when B2 increased (x={self.x:.4f}, x'={x_prime:.4f})")
            
            # Also check that ΔQ' < ΔQ (since ΔQ ∝ 1/B2)
            if DeltaQ_prime > self.DeltaQ:
                logs.append(f"Monotonicity check failed: ΔQ increased when B2 increased")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
        
        # Restore original value (just for safety)
        self.B2 = original_B2
        
        return logs

    def run_all_tests(self):
        all_logs = []
        
        # Run each test
        invariants_logs = self.test_invariants()
        if invariants_logs:
            all_logs.append("Invariants test failures:")
            all_logs.extend(invariants_logs)
        
        trace_logs = self.test_trace_consistency()
        if trace_logs:
            all_logs.append("Trace consistency test failures:")
            all_logs.extend(trace_logs)
        
        monotonicity_logs = self.test_monotonicity()
        if monotonicity_logs:
            all_logs.append("Monotonicity test failures:")
            all_logs.extend(monotonicity_logs)
        
        success = len(all_logs) == 0
        return success, all_logs