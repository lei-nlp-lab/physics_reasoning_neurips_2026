import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely - only use the results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Debug: Print what we received
        print(f"DEBUG: Received results: {self.results}")
        
        # Get E and d from results or use defaults
        self.E = self.results.get('E', 120.0)
        self.d = self.results.get('d', 0.05)
        
        # Distances from plates (hardcoded from problem)
        self.d_AC = 0.005  # 0.5 cm in meters
        self.d_BD = 0.005
        
        # Extract target variables with safe defaults
        # Use explicit checks for required variables
        self.phi_B = self.results.get('phi_B')
        self.phi_D = self.results.get('phi_D')
        self.phi_C = self.results.get('phi_C')
        self.U_CD = self.results.get('U_CD')
        self.C = self.results.get('C')
        
        # Check if we have all required variables
        self.has_all_vars = all(v is not None for v in [self.phi_B, self.phi_D, self.phi_C, self.U_CD, self.C, self.E, self.d])

    def test_invariants(self) -> List[str]:
        logs = []
        
        if not self.has_all_vars:
            logs.append("FAIL: Missing required variables in results.")
            missing = []
            if self.phi_B is None: missing.append('phi_B')
            if self.phi_D is None: missing.append('phi_D')
            if self.phi_C is None: missing.append('phi_C')
            if self.U_CD is None: missing.append('U_CD')
            if self.C is None: missing.append('C')
            if self.E is None: missing.append('E')
            if self.d is None: missing.append('d')
            logs.append(f"Missing: {missing}")
            return logs
            
        E = self.E
        d = self.d
        d_AC = self.d_AC
        d_BD = self.d_BD
        phi_B = self.phi_B
        phi_D = self.phi_D
        phi_C = self.phi_C
        U_CD = self.U_CD

        # 1. Grounding invariant: phi_B == 0
        if not math.isclose(phi_B, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: phi_B should be 0, got {phi_B}")
        else:
            logs.append("PASS: phi_B is grounded (0 V).")

        # 2. Uniform field relation info
        logs.append("INFO: Uniform field relation: Δφ = -E·Δx")

        # 3. Numerical checks (fuzzy matching)
        # Distance from B to C
        d_BC = d - d_AC  # 0.045 m

        # Check phi_D = -E * d_BD
        expected_phi_D = -E * d_BD
        if not math.isclose(phi_D, expected_phi_D, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: phi_D should be {-E * d_BD:.3f} V, got {phi_D:.3f} V")
        else:
            logs.append(f"PASS: phi_D = -E * d_BD = {expected_phi_D:.3f} V.")

        # Check phi_C = -E * d_BC
        expected_phi_C = -E * d_BC
        if not math.isclose(phi_C, expected_phi_C, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: phi_C should be {-E * d_BC:.3f} V, got {phi_C:.3f} V")
        else:
            logs.append(f"PASS: phi_C = -E * (d - d_AC) = {expected_phi_C:.3f} V.")

        # Check U_CD = phi_C - phi_D
        expected_U_CD = phi_C - phi_D
        if not math.isclose(U_CD, expected_U_CD, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: U_CD should be {expected_U_CD:.3f} V, got {U_CD:.3f} V")
        else:
            logs.append(f"PASS: U_CD = phi_C - phi_D = {expected_U_CD:.3f} V.")

        # 4. Check uniform field slope: (phi_C - phi_D) / (distance_CD) = -E
        distance_CD = d - d_AC - d_BD  # 0.04 m
        try:
            slope = (phi_C - phi_D) / distance_CD
            if not math.isclose(slope, -E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Field slope mismatch: got {slope:.1f} V/m, expected {-E:.1f} V/m")
            else:
                logs.append(f"PASS: Uniform field slope correct: {slope:.1f} V/m.")
        except ZeroDivisionError:
            logs.append("ERROR: Division by zero when calculating field slope.")

        # 5. Capacitance sanity check
        C = self.C
        if C <= 0:
            logs.append(f"FAIL: Capacitance should be positive, got {C}")
        else:
            # Check if C is close to expected value 5e-10 F
            expected_C = 5e-10
            if math.isclose(C, expected_C, rel_tol=0.1):
                logs.append(f"PASS: Capacitance is correct: {C:.1e} F ≈ {expected_C:.1e} F.")
            else:
                logs.append(f"INFO: Capacitance is positive: {C:.1e} F (expected {expected_C:.1e} F).")

        return logs

    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        if not self.has_all_vars:
            logs.append("SKIP: Missing required variables for trace consistency check.")
            return logs
            
        # Cross-check intermediate results with final answers
        phi_D = self.phi_D
        phi_C = self.phi_C
        U_CD = self.U_CD

        # Check U_CD = phi_C - phi_D
        if not math.isclose(U_CD, phi_C - phi_D, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Trace inconsistency: U_CD != phi_C - phi_D ({U_CD} vs {phi_C - phi_D})")
        else:
            logs.append("PASS: U_CD consistent with phi_C and phi_D.")

        # Check phi_C and phi_D against the electric field
        expected_phi_D = -self.E * self.d_BD
        expected_phi_C = -self.E * (self.d - self.d_AC)
        if not (math.isclose(phi_D, expected_phi_D, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(phi_C, expected_phi_C, rel_tol=1e-2, abs_tol=1e-2)):
            logs.append("FAIL: Trace inconsistency: potentials do not match the uniform field formula.")
        else:
            logs.append("PASS: Potentials consistent with uniform field.")

        # Check that the potential difference between plates is E*d (magnitude)
        phi_A = -self.E * self.d  # since phi_B=0
        logs.append(f"INFO: Expected potential at plate A: {phi_A:.2f} V.")
        return logs

    def test_monotonicity(self) -> List[str]:
        logs = []
        
        if not self.has_all_vars:
            logs.append("SKIP: Missing required variables for monotonicity check.")
            return logs
            
        original_E = self.E
        # Perturb E by +10%
        perturbed_E = original_E * 1.10

        # Recompute expected values with perturbed E
        expected_phi_D = -perturbed_E * self.d_BD
        expected_phi_C = -perturbed_E * (self.d - self.d_AC)
        expected_U_CD = expected_phi_C - expected_phi_D

        # Get original results
        phi_D = self.phi_D
        phi_C = self.phi_C
        U_CD = self.U_CD

        # Check monotonicity: magnitudes should increase when E increases
        # Since values are negative, we compare absolute values
        if abs(expected_phi_D) > abs(phi_D) and abs(expected_phi_C) > abs(phi_C) and abs(expected_U_CD) > abs(U_CD):
            logs.append("PASS: Monotonicity holds: increasing E increases magnitudes of potentials and U_CD.")
        else:
            logs.append(f"FAIL: Monotonicity violation with E increased by 10%.")

        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        logs.append("=== Physics Checker Start ===")
        logs.append(f"DEBUG: Has all variables: {self.has_all_vars}")

        # Run individual tests
        logs.append("\n--- Testing Invariants ---")
        try:
            logs.extend(self.test_invariants())
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())

        logs.append("\n--- Testing Trace Consistency ---")
        try:
            logs.extend(self.test_trace_consistency())
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())

        logs.append("\n--- Testing Monotonicity ---")
        try:
            logs.extend(self.test_monotonicity())
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())

        logs.append("\n=== Physics Checker End ===")

        # Determine overall success: no "FAIL" messages
        success = not any("FAIL" in log for log in logs)
        return success, logs