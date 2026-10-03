import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.N = getattr(solution_instance, 'N', 100)
        self.S = getattr(solution_instance, 'S', 0.4)  # Area of region D
        self.r = getattr(solution_instance, 'r', 2.0)
        self.R1 = getattr(solution_instance, 'R1', 2.0)
        self.R2 = getattr(solution_instance, 'R2', 6.0)
        self.C = getattr(solution_instance, 'C', 5e-6)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.E = getattr(solution_instance, 'E', results.get('E', 0.0))
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.U = getattr(solution_instance, 'U', results.get('U', 0.0))
        self.Q = getattr(solution_instance, 'Q', results.get('Q', 0.0))
        
        # Target answers for comparison
        self.target_I = 0.8
        self.target_Q = 2.4e-5
        
        # Physical constants (if any)
        self.dt = getattr(solution_instance, 'dt', 1.0)  # Default Δt if not specified
        
        # Store original C for monotonicity test
        self.original_C = self.C
        
        # Logs
        self.logs = []
    
    def log(self, message, success=None):
        """Helper to log messages"""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_invariants(self):
        """Check fundamental physics laws: Ohm's Law, Faraday's Law, Energy Conservation"""
        try:
            # 1. Ohm's Law for entire circuit: E = I*(R1 + R2 + r)
            total_R = self.R1 + self.R2 + self.r
            ohm_check = math.isclose(self.E, self.I * total_R, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Ohm's Law: E = {self.E} ≈ I*(R1+R2+r) = {self.I}*{total_R} = {self.I*total_R}", ohm_check)
            
            # 2. Ohm's Law for R2: U = I*R2
            ohm_R2_check = math.isclose(self.U, self.I * self.R2, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Ohm's Law for R2: U = {self.U} ≈ I*R2 = {self.I}*{self.R2} = {self.I*self.R2}", ohm_R2_check)
            
            # 3. Capacitance formula: Q = C*U
            cap_check = math.isclose(self.Q, self.original_C * self.U, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Capacitance: Q = {self.Q} ≈ C*U = {self.original_C}*{self.U} = {self.original_C*self.U}", cap_check)
            
            # 4. Energy Conservation: Power supplied = Power dissipated
            # Power from induced EMF: P_emf = E*I
            # Power dissipated in resistors: P_diss = I²*(R1+R2+r)
            P_emf = self.E * self.I
            P_diss = self.I**2 * total_R
            energy_check = math.isclose(P_emf, P_diss, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Energy: P_emf = E*I = {P_emf} ≈ I²*(R1+R2+r) = {P_diss}", energy_check)
            
            return all([ohm_check, ohm_R2_check, cap_check, energy_check])
            
        except Exception as e:
            self.log(f"Error in test_invariants: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Step 1: Faraday's Law
            # From KB: emf_avg = -N * ΔΦ/Δt
            # We don't have ΔΦ/Δt, but we can check consistency with given steps
            
            # Step 2: I = E / (R1 + R2 + r)
            calc_I = self.E / (self.R1 + self.R2 + self.r)
            step2_check = math.isclose(self.I, calc_I, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Step 2: I = {self.I} ≈ E/(R1+R2+r) = {self.E}/{self.R1+self.R2+self.r} = {calc_I}", step2_check)
            
            # Step 3: U = I*R2
            calc_U = self.I * self.R2
            step3_check = math.isclose(self.U, calc_U, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Step 3: U = {self.U} ≈ I*R2 = {self.I}*{self.R2} = {calc_U}", step3_check)
            
            # Step 4: Q = C*U
            calc_Q = self.original_C * self.U
            step4_check = math.isclose(self.Q, calc_Q, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Step 4: Q = {self.Q} ≈ C*U = {self.original_C}*{self.U} = {calc_Q}", step4_check)
            
            # Check against target answers
            target_I_check = math.isclose(self.I, self.target_I, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Target I: {self.I} ≈ {self.target_I}", target_I_check)
            
            target_Q_check = math.isclose(self.Q, self.target_Q, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Target Q: {self.Q} ≈ {self.target_Q}", target_Q_check)
            
            return all([step2_check, step3_check, step4_check, target_I_check, target_Q_check])
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input parameter and check logical trend"""
        try:
            # Save original values
            orig_R2 = self.R2
            orig_I = self.I
            orig_U = self.U
            orig_Q = self.Q
            
            # Perturb R2 by +10%
            perturbed_R2 = self.R2 * 1.1
            
            # Recalculate using formulas (simulating the effect)
            new_I = self.E / (self.R1 + perturbed_R2 + self.r)
            new_U = new_I * perturbed_R2
            new_Q = self.original_C * new_U
            
            # Expected trends from Ohm's Law:
            # 1. I should decrease when R2 increases (since total resistance increases)
            # 2. U across R2 might increase or decrease? Let's check:
            #    U = E * R2/(R1+R2+r) = E/(1 + (R1+r)/R2)
            #    As R2 increases, U increases toward E
            # 3. Q = C*U should follow U's trend
            
            trend1 = new_I < orig_I  # Current decreases
            trend2 = new_U > orig_U  # Voltage across R2 increases
            trend3 = new_Q > orig_Q  # Charge increases
            
            self.log(f"After R2↑ {orig_R2}→{perturbed_R2}: I {orig_I}→{new_I} (decrease)", trend1)
            self.log(f"After R2↑ {orig_R2}→{perturbed_R2}: U {orig_U}→{new_U} (increase)", trend2)
            self.log(f"After R2↑ {orig_R2}→{perturbed_R2}: Q {orig_Q}→{new_Q} (increase)", trend3)
            
            return all([trend1, trend2, trend3])
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        
        self.log("=== PhysicsChecker Starting ===")
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall = all([test1, test2, test3])
        
        self.log(f"=== PhysicsChecker Complete: {'ALL PASS' if overall else 'SOME FAILED'} ===")
        
        return overall, self.logs