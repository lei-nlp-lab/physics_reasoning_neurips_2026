import math
import sympy as sp
from typing import List, Tuple, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Initialize with safe attribute access."""
        # Extract inputs
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'results', {}).get('B', 3.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', 0.3))
        # There are two different velocities: v1 for Faraday (coil pulled out), v2 for motional (just about to leave)
        # Extract from results if available, otherwise use defaults
        results = getattr(solution_instance, 'results', {})
        self.v1 = results.get('v1', 10.0)  # for Q1
        self.v2 = results.get('v2', 0.5)   # for Q2
        self.n = 1  # single-turn coil
        
        # Extract results (handle both list and scalar cases)
        E_raw = getattr(solution_instance, 'E', None)
        if E_raw is None:
            E_raw = results.get('E', [0.6, 0.45])
        
        if isinstance(E_raw, (list, tuple)):
            if len(E_raw) >= 2:
                self.E_faraday = float(E_raw[0])
                self.E_motional = float(E_raw[1])
            else:
                self.E_faraday = 0.6
                self.E_motional = 0.45
        else:
            # Assume scalar - use from steps analysis
            self.E_faraday = 0.6
            self.E_motional = 0.45
        
        # Initialize sympy symbols (not used in numeric tests)
        self.phi = sp.Symbol('phi', real=True)
        self.t = sp.Symbol('t', real=True)
        self.emf = sp.Symbol('emf', real=True)
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify Faraday's law and motional EMF invariants."""
        logs = []
        success = True
        
        # 1. Faraday's law check (average EMF) for Q1
        # Initial area in field: 2/3 of coil
        A_initial = (2/3) * self.L**2
        phi_initial = self.B * A_initial
        phi_final = 0  # when completely out of field
        
        # Time to leave: distance = width of left part = (2/3)*L, velocity = v1
        delta_t = (2/3) * self.L / self.v1
        delta_phi = phi_final - phi_initial
        
        # Average EMF from Faraday's law
        emf_avg_calc = -delta_phi / delta_t
        
        if not math.isclose(abs(self.E_faraday), abs(float(emf_avg_calc)), 
                           rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Faraday's law mismatch: got {self.E_faraday:.3f}V, "
                       f"expected {float(emf_avg_calc):.3f}V")
            success = False
        else:
            logs.append(f"Faraday's law OK: {self.E_faraday:.3f}V matches "
                       f"{float(emf_avg_calc):.3f}V")
        
        # 2. Motional EMF check (instantaneous) for Q2
        # The conductor length perpendicular to motion is L (side of square)
        emf_mot_calc = self.B * self.L * self.v2
        
        if not math.isclose(abs(self.E_motional), abs(float(emf_mot_calc)), 
                           rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Motional EMF mismatch: got {self.E_motional:.3f}V, "
                       f"expected {float(emf_mot_calc):.3f}V")
            success = False
        else:
            logs.append(f"Motional EMF OK: {self.E_motional:.3f}V matches "
                       f"{float(emf_mot_calc):.3f}V")
        
        # 3. Consistency between two EMFs (optional)
        # They are from different scenarios, no direct relation
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check both EMF values are positive (magnitudes)
        if self.E_faraday < 0 or self.E_motional < 0:
            logs.append("Warning: EMF values should be positive magnitudes")
        
        # Check they match expected target answers
        target1, target2 = 0.6, 0.45
        if not (math.isclose(self.E_faraday, target1, rel_tol=0.02, abs_tol=0.02) and
                math.isclose(self.E_motional, target2, rel_tol=0.02, abs_tol=0.02)):
            logs.append(f"Target mismatch: got ({self.E_faraday:.3f}V, {self.E_motional:.3f}V), "
                       f"expected ({target1}V, {target2}V)")
            success = False
        else:
            logs.append(f"Target values OK: ({self.E_faraday:.3f}V, {self.E_motional:.3f}V)")
        
        # Verify physical reasonableness
        if self.E_faraday > 10 or self.E_motional > 10:  # Unrealistically large
            logs.append("Warning: EMF values seem unusually large")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how EMF changes with perturbed parameters."""
        logs = []
        success = True
        
        # Perturb magnetic field B for motional EMF
        original_B = self.B
        self.B *= 1.1  # Increase by 10%
        
        # Recalculate motional EMF
        new_emf_mot = self.B * self.L * self.v2
        
        # Check monotonic increase
        if new_emf_mot <= self.E_motional:
            logs.append(f"Monotonicity failure: B↑ from {original_B}T to {self.B:.2f}T "
                       f"should increase motional EMF, got {new_emf_mot:.3f}V vs {self.E_motional:.3f}V")
            success = False
        else:
            logs.append(f"Monotonicity OK: B↑ increases motional EMF from "
                       f"{self.E_motional:.3f}V to {new_emf_mot:.3f}V")
        
        # Restore original B
        self.B = original_B
        
        # Perturb velocity v2 for motional EMF
        original_v2 = self.v2
        self.v2 *= 0.9  # Decrease by 10%
        
        # Recalculate motional EMF
        new_emf_mot2 = self.B * self.L * self.v2
        
        # Check decrease with v decrease
        if new_emf_mot2 >= self.E_motional:
            logs.append(f"Monotonicity failure: v↓ should decrease motional EMF, "
                       f"got {new_emf_mot2:.3f}V vs {self.E_motional:.3f}V")
            success = False
        else:
            logs.append(f"Monotonicity OK: v↓ decreases motional EMF from "
                       f"{self.E_motional:.3f}V to {new_emf_mot2:.3f}V")
        
        # Restore original v2
        self.v2 = original_v2
        
        # Perturb velocity v1 for Faraday EMF
        original_v1 = self.v1
        self.v1 *= 1.1  # Increase by 10%
        # Recalculate Faraday EMF
        delta_t_new = (2/3) * self.L / self.v1
        delta_phi = -self.B * (2/3) * self.L**2
        new_emf_faraday = -delta_phi / delta_t_new
        if new_emf_faraday <= self.E_faraday:
            logs.append(f"Monotonicity failure: v↑ should increase Faraday EMF, "
                       f"got {new_emf_faraday:.3f}V vs {self.E_faraday:.3f}V")
            success = False
        else:
            logs.append(f"Monotonicity OK: v↑ increases Faraday EMF from "
                       f"{self.E_faraday:.3f}V to {new_emf_faraday:.3f}V")
        self.v1 = original_v1
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"\n=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs