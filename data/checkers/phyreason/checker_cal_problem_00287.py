import math
import sympy as sp
from typing import List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        Never directly access solution_instance.attr - use getattr with fallback to results dict.
        """
        # Extract inputs with safe defaults
        self.R = getattr(solution_instance, 'R', 0.45)
        self.M = getattr(solution_instance, 'M', 1.0)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.h = getattr(solution_instance, 'h', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results using the provided key mapping
        self.results = {}
        result_keys = [
            '1/2*mv_B2', 'x', 'a_1', "t''", 't_1', 'v_block', 't_2', "N'", 
            'W_f', 'a_2', 'v', 'v_car', 'N-mg', 'v_object', 'v_B', 'N'
        ]
        
        # Safely extract each result with fallback to 0.0
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, solution_instance.results.get(key, 0.0))
        
        # Target answers (numerical values only)
        self.targets = {
            'N': 30.0,
            'v_block': 1.0,
            'x': 0.2
        }
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self) -> List[str]:
        """Test conservation laws and Newton's laws based on KB."""
        logs = []
        
        try:
            # 1. Conservation of Mechanical Energy (A to B)
            # Initial potential energy at A: m*g*R
            # Kinetic energy at B: 1/2*m*v_B^2
            v_B = self.results.get('v_B', 0.0)
            ke_B = 0.5 * self.m * v_B**2
            pe_A = self.m * self.g * self.R
            
            if not math.isclose(ke_B, pe_A, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Energy conservation violation: m*g*R = {pe_A:.3f}, 1/2*m*v_B^2 = {ke_B:.3f}")
            else:
                logs.append("✓ Energy conservation (A→B) satisfied")
            
            # 2. Newton's Second Law at B
            # N - mg = m*v_B^2/R
            N = self.results.get('N', 0.0)
            lhs = N - self.m * self.g
            rhs = self.m * v_B**2 / self.R
            
            if not math.isclose(lhs, rhs, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Newton's 2nd Law violation at B: N - mg = {lhs:.3f}, m*v_B^2/R = {rhs:.3f}")
            else:
                logs.append("✓ Newton's 2nd Law at B satisfied")
            
            # 3. Newton's Third Law (N' should equal N)
            N_prime = self.results.get("N'", 0.0)
            if not math.isclose(N_prime, N, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Newton's 3rd Law violation: N' = {N_prime:.3f}, N = {N:.3f}")
            else:
                logs.append("✓ Newton's 3rd Law (N'=N) satisfied")
            
            # 4. Work-Energy Theorem for flatcar segment
            # Initial KE at B + Work_friction = Final KE when leaving flatcar
            W_f = self.results.get('W_f', 0.0)
            v = self.results.get('v', 0.0)  # Velocity when leaving flatcar
            ke_initial = 0.5 * self.m * v_B**2
            ke_final = 0.5 * self.m * v**2
            
            if not math.isclose(ke_initial + W_f, ke_final, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Work-Energy violation: KE_B + W_f = {ke_initial + W_f:.3f}, KE_final = {ke_final:.3f}")
            else:
                logs.append("✓ Work-Energy Theorem (flatcar) satisfied")
        
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Check that v_B from energy matches kinematics usage
            v_B_calc = math.sqrt(2 * self.g * self.R)
            v_B_given = self.results.get('v_B', 0.0)
            
            if not math.isclose(v_B_calc, v_B_given, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"v_B inconsistency: sqrt(2gR) = {v_B_calc:.3f}, given = {v_B_given:.3f}")
            else:
                logs.append("✓ v_B consistent with energy conservation")
            
            # Check accelerations a_1 and a_2
            # From steps: a1 = μ*g, a2 = μ*m*g/M
            # Since μ isn't given, check if a1 = a2 (given m=M=1)
            a1 = self.results.get('a_1', 0.0)
            a2 = self.results.get('a_2', 0.0)
            
            if not math.isclose(a1, a2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Acceleration inconsistency: a1 = {a1:.3f}, a2 = {a2:.3f}")
            else:
                logs.append("✓ Accelerations a1 and a2 consistent (given m=M)")
            
            # Check that t_2 produces correct velocities
            t2 = self.results.get('t_2', 0.0)
            v_block_calc = v_B_given - a1 * t2
            v_block_given = self.results.get('v_block', 0.0)
            
            if not math.isclose(v_block_calc, v_block_given, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"v_block inconsistency: v_B - a1*t2 = {v_block_calc:.3f}, given = {v_block_given:.3f}")
            else:
                logs.append("✓ v_block consistent with kinematics")
            
            # Check projectile time t'
            t_prime_calc = math.sqrt(2 * self.h / self.g)
            t_prime_given = self.results.get("t''", 0.0)
            
            if not math.isclose(t_prime_calc, t_prime_given, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"t' inconsistency: sqrt(2h/g) = {t_prime_calc:.3f}, given = {t_prime_given:.3f}")
            else:
                logs.append("✓ Projectile time t' consistent")
        
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        """
        Perturb mass m and check trends.
        Important: Restore original values after perturbation.
        """
        logs = []
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            self.m = original_m * 1.1
            
            # Recalculate expected values using physics formulas
            # 1. Normal force at B should scale linearly with m
            v_B_new = math.sqrt(2 * self.g * self.R)  # Same since independent of m
            N_new_calc = self.m * self.g + self.m * v_B_new**2 / self.R
            N_new_expected = self.targets['N'] * (self.m / original_m)
            
            if not math.isclose(N_new_calc, N_new_expected, rel_tol=0.05, abs_tol=0.1):
                logs.append(f"Monotonicity issue: N doesn't scale linearly with m. Expected ~{N_new_expected:.3f}, got {N_new_calc:.3f}")
            else:
                logs.append("✓ N scales linearly with m")
            
            # 2. Final velocity when leaving flatcar (simplified check)
            # For same friction coefficient, with increased m, both friction force and inertia increase
            # Hard to predict exact trend without full system, so we skip quantitative check
            logs.append("Note: v_block monotonicity check requires full system solution")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
        finally:
            # RESTORE original mass
            self.m = original_m
        
        return logs

    def test_target_answers(self) -> List[str]:
        """Check if final answers match target values."""
        logs = []
        
        try:
            N = self.results.get('N', 0.0)
            v_block = self.results.get('v_block', 0.0)
            x = self.results.get('x', 0.0)
            
            if not math.isclose(N, self.targets['N'], rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Target mismatch: N = {N:.3f}, expected {self.targets['N']}")
            else:
                logs.append(f"✓ N matches target: {N:.3f} N")
            
            if not math.isclose(v_block, self.targets['v_block'], rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Target mismatch: v_block = {v_block:.3f}, expected {self.targets['v_block']}")
            else:
                logs.append(f"✓ v_block matches target: {v_block:.3f} m/s")
            
            if not math.isclose(x, self.targets['x'], rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Target mismatch: x = {x:.3f}, expected {self.targets['x']}")
            else:
                logs.append(f"✓ x matches target: {x:.3f} m")
        
        except Exception as e:
            logs.append(f"Error in test_target_answers: {e}")
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        all_logs = []
        
        all_logs.append("=== PhysicsChecker Report ===")
        
        # Run tests
        all_logs.append("\n1. Target Answers Check:")
        all_logs.extend(self.test_target_answers())
        
        all_logs.append("\n2. Physics Invariants:")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n3. Trace Consistency:")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n4. Monotonicity Check:")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith(("Energy conservation violation", 
                               "Newton's", 
                               "Work-Energy violation",
                               "inconsistency",
                               "Target mismatch",
                               "Monotonicity issue",
                               "Error in")):
                if not log.startswith("✓"):
                    success = False
        
        return success, all_logs