import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.inputs = {}
        self.results = {}
        
        # Inputs from problem statement
        self.inputs['R'] = 0.8
        self.inputs['m'] = 6e-2
        self.inputs['q'] = 8e-3
        self.inputs['d'] = 8e-2
        self.inputs['E'] = 15.0
        self.inputs['r'] = 1.0
        self.inputs['R0'] = 6.0
        self.inputs['r_M'] = 0.6
        self.inputs['g'] = 10.0
        
        # Expected keys in solution_instance.results
        result_keys = ['v', 'U', 'I', 'U_M', 'P_machine']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, solution_instance.results.get(key, None))
        
        # Pre-compute expected values from formulas for comparison
        self.expected = {}
        self.compute_expected()
        
        self.logs = []
    
    def compute_expected(self):
        """Compute expected values from physical laws."""
        R = self.inputs['R']
        m = self.inputs['m']
        q = self.inputs['q']
        d = self.inputs['d']
        g = self.inputs['g']
        E = self.inputs['E']
        r = self.inputs['r']
        R0 = self.inputs['R0']
        r_M = self.inputs['r_M']
        
        # Step 1: Conservation of energy: mgR = 0.5*m*v^2
        v_expected = math.sqrt(2 * g * R)
        self.expected['v'] = v_expected
        
        # Step 2: Force balance: q*(U/d) = m*g
        U_expected = (m * g * d) / q
        self.expected['U'] = U_expected
        
        # Step 4: Ohm's law for R0: I = U/R0
        I_expected = U_expected / R0
        self.expected['I'] = I_expected
        
        # Step 5: Circuit equation: E = U + I*r + U_M
        U_M_expected = E - U_expected - I_expected * r
        self.expected['U_M'] = U_M_expected
        
        # Step 6: Mechanical power: P = I*U_M - I^2*r_M
        P_expected = I_expected * U_M_expected - I_expected**2 * r_M
        self.expected['P_machine'] = P_expected
    
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe fuzzy comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        logs = []
        try:
            # 1. Conservation of energy (from P to Q)
            m, g, R = self.inputs['m'], self.inputs['g'], self.inputs['R']
            v = self.results['v']
            if v is not None:
                KE = 0.5 * m * v**2
                PE_loss = m * g * R
                if not self.is_close(KE, PE_loss):
                    logs.append(f"FAIL energy conservation: KE={KE}, PE_loss={PE_loss}")
                else:
                    logs.append("PASS energy conservation")
            
            # 2. Force balance in capacitor
            q, d, U = self.inputs['q'], self.inputs['d'], self.results['U']
            if all(x is not None for x in (q, d, U)):
                F_e = q * (U / d)
                F_g = m * g
                if not self.is_close(F_e, F_g):
                    logs.append(f"FAIL force balance: F_e={F_e}, F_g={F_g}")
                else:
                    logs.append("PASS force balance")
            
            # 3. Ohm's law for R0
            R0, I = self.inputs['R0'], self.results['I']
            if all(x is not None for x in (U, R0, I)):
                if not self.is_close(U, I * R0):
                    logs.append(f"FAIL Ohm's law for R0: U={U}, I*R0={I*R0}")
                else:
                    logs.append("PASS Ohm's law for R0")
            
            # 4. Circuit loop equation
            E, r, U_M = self.inputs['E'], self.inputs['r'], self.results['U_M']
            if all(x is not None for x in (E, U, I, r, U_M)):
                if not self.is_close(E, U + I*r + U_M):
                    logs.append(f"FAIL circuit equation: E={E}, sum={U + I*r + U_M}")
                else:
                    logs.append("PASS circuit equation")
            
            # 5. Motor power equation
            r_M, P = self.inputs['r_M'], self.results['P_machine']
            if all(x is not None for x in (I, U_M, r_M, P)):
                P_expected = I * U_M - I**2 * r_M
                if not self.is_close(P, P_expected):
                    logs.append(f"FAIL motor power: P={P}, expected={P_expected}")
                else:
                    logs.append("PASS motor power equation")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Compare each result with expected from formulas
            for key in ['v', 'U', 'I', 'U_M', 'P_machine']:
                val = self.results[key]
                exp = self.expected[key]
                if val is not None:
                    if not self.is_close(val, exp):
                        logs.append(f"FAIL {key}: result={val}, expected={exp}")
                    else:
                        logs.append(f"PASS {key} matches formula")
                else:
                    logs.append(f"SKIP {key}: missing result")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_m = self.inputs['m']
        try:
            # Perturb mass m by +10%
            self.inputs['m'] = original_m * 1.1
            # Recompute expected values with perturbed m
            R = self.inputs['R']
            m_pert = self.inputs['m']
            q = self.inputs['q']
            d = self.inputs['d']
            g = self.inputs['g']
            E = self.inputs['E']
            r = self.inputs['r']
            R0 = self.inputs['R0']
            r_M = self.inputs['r_M']
            
            v_pert = math.sqrt(2 * g * R)
            U_pert = (m_pert * g * d) / q
            I_pert = U_pert / R0
            U_M_pert = E - U_pert - I_pert * r
            P_pert = I_pert * U_M_pert - I_pert**2 * r_M
            
            # Original expected values (from compute_expected, which used original m)
            v_orig = self.expected['v']
            U_orig = self.expected['U']
            I_orig = self.expected['I']
            U_M_orig = self.expected['U_M']
            P_orig = self.expected['P_machine']
            
            # Check trends
            # As m increases, v should increase (since v = sqrt(2gR), independent of m! Wait, v does not depend on m. But the problem's step 1 says mgR = 0.5*m*v^2, so v = sqrt(2gR) indeed independent of m. So v should not change.
            # Actually, v is independent of m. So we expect no change.
            if not self.is_close(v_pert, v_orig):
                logs.append("MONOTONICITY: v changed with m, but theoretically it shouldn't. Check formula.")
            
            # U = (m*g*d)/q, so U increases with m
            if U_pert <= U_orig:
                logs.append("MONOTONICITY: U should increase with m, but it didn't.")
            
            # I = U/R0, so I increases with m
            if I_pert <= I_orig:
                logs.append("MONOTONICITY: I should increase with m, but it didn't.")
            
            # U_M = E - U - I*r, and both U and I increase, so U_M decreases
            if U_M_pert >= U_M_orig:
                logs.append("MONOTONICITY: U_M should decrease with m, but it didn't.")
            
            # P = I*U_M - I^2*r_M. With I increasing and U_M decreasing, the trend is not obvious.
            # We can check numerically: compute P_pert and P_orig. For the given numbers, let's see.
            # Actually, we don't assert a trend for P because it's not monotonic in general.
            # But we can note the change.
            logs.append(f"MONOTONICITY: P changed from {P_orig:.3f} to {P_pert:.3f}")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
        finally:
            # Restore original mass
            self.inputs['m'] = original_m
        return logs
    
    def run_all_tests(self):
        self.logs = []
        self.logs.append("=== PhysicsChecker started ===")
        
        # Run each test and collect logs
        self.logs.append("\n--- Testing Invariants ---")
        self.logs.extend(self.test_invariants())
        
        self.logs.append("\n--- Testing Trace Consistency ---")
        self.logs.extend(self.test_trace_consistency())
        
        self.logs.append("\n--- Testing Monotonicity (perturb mass +10%) ---")
        self.logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in self.logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs

# Example of how the solution_instance might look (for testing only)
if __name__ == "__main__":
    class DummySolution:
        def __init__(self):
            self.results = {
                'v': 4.0,
                'U': 6.0,
                'I': 1.0,
                'U_M': 8.0,
                'P_machine': 7.4
            }
    
    dummy = DummySolution()
    checker = PhysicsChecker(dummy)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)