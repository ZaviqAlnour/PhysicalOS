"""
Physical Intelligence Layer - Center of Mass and Body Dynamics
Calculates biomechanical properties like CoM, balance, and physical metrics.
"""

import numpy as np


# Approximate body segment masses (% of total body mass)
# Based on biomechanics literature (Winter, 2009)
SEGMENT_MASS_RATIOS = {
    'HEAD': 0.081,
    'TORSO': 0.497,  # Trunk (thorax + abdomen)
    'UPPER_ARM_L': 0.028,
    'UPPER_ARM_R': 0.028,
    'FOREARM_L': 0.016,
    'FOREARM_R': 0.016,
    'HAND_L': 0.006,
    'HAND_R': 0.006,
    'THIGH_L': 0.100,
    'THIGH_R': 0.100,
    'SHANK_L': 0.0465,
    'SHANK_R': 0.0465,
    'FOOT_L': 0.0145,
    'FOOT_R': 0.0145,
}

# Segment CoM locations (% along segment from proximal end)
SEGMENT_COM_RATIOS = {
    'HEAD': 0.50,
    'TORSO': 0.50,
    'UPPER_ARM': 0.436,
    'FOREARM': 0.430,
    'HAND': 0.506,
    'THIGH': 0.433,
    'SHANK': 0.433,
    'FOOT': 0.50,
}


class BodySegment:
    """
    Represents a body segment with mass and position.
    """
    
    def __init__(self, name, proximal_joint, distal_joint, mass_ratio):
        """
        Initialize body segment.
        
        Args:
            name: Segment name
            proximal_joint: Joint at proximal end
            distal_joint: Joint at distal end
            mass_ratio: Mass as fraction of total body mass
        """
        self.name = name
        self.proximal_joint = proximal_joint
        self.distal_joint = distal_joint
        self.mass_ratio = mass_ratio
    
    def get_com_position(self, joint_positions, com_ratio=0.5):
        """
        Calculate center of mass position for this segment.
        
        Args:
            joint_positions: Dict of joint positions
            com_ratio: CoM location ratio along segment
            
        Returns:
            numpy array: 3D CoM position or None
        """
        if self.proximal_joint not in joint_positions or \
           self.distal_joint not in joint_positions:
            return None
        
        prox_pos = joint_positions[self.proximal_joint]
        dist_pos = joint_positions[self.distal_joint]
        
        # CoM is at ratio point along segment
        com_pos = prox_pos + com_ratio * (dist_pos - prox_pos)
        
        return com_pos


class CenterOfMassCalculator:
    """
    Calculate whole-body center of mass from joint positions.
    """
    
    def __init__(self, total_body_mass=70.0):
        """
        Initialize CoM calculator.
        
        Args:
            total_body_mass: Total body mass in kg (default: 70kg)
        """
        self.total_mass = total_body_mass
        self.segments = self._define_segments()
    
    def _define_segments(self):
        """
        Define all body segments.
        
        Returns:
            list: Body segment objects
        """
        segments = [
            # Head & Torso
            BodySegment('HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
                       SEGMENT_MASS_RATIOS['HEAD']),
            BodySegment('TORSO', 'MID_HIP', 'LEFT_SHOULDER',
                       SEGMENT_MASS_RATIOS['TORSO'] / 2),
            BodySegment('TORSO', 'MID_HIP', 'RIGHT_SHOULDER',
                       SEGMENT_MASS_RATIOS['TORSO'] / 2),
            
            # Left arm
            BodySegment('UPPER_ARM_L', 'LEFT_SHOULDER', 'LEFT_ELBOW',
                       SEGMENT_MASS_RATIOS['UPPER_ARM_L']),
            BodySegment('FOREARM_L', 'LEFT_ELBOW', 'LEFT_WRIST',
                       SEGMENT_MASS_RATIOS['FOREARM_L']),
            BodySegment('HAND_L', 'LEFT_WRIST', 'LEFT_WRIST',  # Point mass
                       SEGMENT_MASS_RATIOS['HAND_L']),
            
            # Right arm
            BodySegment('UPPER_ARM_R', 'RIGHT_SHOULDER', 'RIGHT_ELBOW',
                       SEGMENT_MASS_RATIOS['UPPER_ARM_R']),
            BodySegment('FOREARM_R', 'RIGHT_ELBOW', 'RIGHT_WRIST',
                       SEGMENT_MASS_RATIOS['FOREARM_R']),
            BodySegment('HAND_R', 'RIGHT_WRIST', 'RIGHT_WRIST',
                       SEGMENT_MASS_RATIOS['HAND_R']),
            
            # Left leg
            BodySegment('THIGH_L', 'LEFT_HIP', 'LEFT_KNEE',
                       SEGMENT_MASS_RATIOS['THIGH_L']),
            BodySegment('SHANK_L', 'LEFT_KNEE', 'LEFT_ANKLE',
                       SEGMENT_MASS_RATIOS['SHANK_L']),
            BodySegment('FOOT_L', 'LEFT_ANKLE', 'LEFT_ANKLE',
                       SEGMENT_MASS_RATIOS['FOOT_L']),
            
            # Right leg
            BodySegment('THIGH_R', 'RIGHT_HIP', 'RIGHT_KNEE',
                       SEGMENT_MASS_RATIOS['THIGH_R']),
            BodySegment('SHANK_R', 'RIGHT_KNEE', 'RIGHT_ANKLE',
                       SEGMENT_MASS_RATIOS['SHANK_R']),
            BodySegment('FOOT_R', 'RIGHT_ANKLE', 'RIGHT_ANKLE',
                       SEGMENT_MASS_RATIOS['FOOT_R']),
        ]
        
        return segments
    
    def calculate_com(self, joint_positions):
        """
        Calculate whole-body center of mass.
        
        Args:
            joint_positions: Dict of {joint_name: [x, y, z]}
            
        Returns:
            numpy array: CoM position [x, y, z] or None
        """
        total_weighted_pos = np.zeros(3)
        total_mass_accounted = 0.0
        
        for segment in self.segments:
            # Get segment CoM
            com_ratio = SEGMENT_COM_RATIOS.get(
                segment.name.split('_')[0], 0.5
            )
            segment_com = segment.get_com_position(joint_positions, com_ratio)
            
            if segment_com is not None:
                segment_mass = segment.mass_ratio * self.total_mass
                total_weighted_pos += segment_com * segment_mass
                total_mass_accounted += segment_mass
        
        if total_mass_accounted < 0.1:  # Less than 10% of body mass tracked
            return None
        
        # Weighted average
        com = total_weighted_pos / total_mass_accounted
        
        return com
    
    def calculate_com_relative_to_hip(self, joint_positions):
        """
        Calculate CoM relative to Mid-Hip (for consistency with other data).
        
        Args:
            joint_positions: Dict of joint positions
            
        Returns:
            numpy array: CoM position relative to Mid-Hip
        """
        com = self.calculate_com(joint_positions)
        
        if com is None:
            return None
        
        # Subtract Mid-Hip position (should already be at origin in our data)
        mid_hip = joint_positions.get('MID_HIP', np.zeros(3))
        
        return com - mid_hip


class BalanceAnalyzer:
    """
    Analyze balance and stability from CoM data.
    """
    
    @staticmethod
    def calculate_base_of_support(left_foot_pos, right_foot_pos):
        """
        Calculate center of base of support (between feet).
        
        Args:
            left_foot_pos: Left ankle position [x, y, z]
            right_foot_pos: Right ankle position [x, y, z]
            
        Returns:
            numpy array: BoS center [x, y, z]
        """
        return (left_foot_pos + right_foot_pos) / 2.0
    
    @staticmethod
    def calculate_com_to_bos_distance(com_pos, bos_center):
        """
        Calculate horizontal distance from CoM to base of support.
        
        Args:
            com_pos: Center of mass position
            bos_center: Base of support center
            
        Returns:
            float: Horizontal distance (stability metric)
        """
        # Only use X and Z (horizontal plane)
        com_horizontal = np.array([com_pos[0], com_pos[2]])
        bos_horizontal = np.array([bos_center[0], bos_center[2]])
        
        distance = np.linalg.norm(com_horizontal - bos_horizontal)
        
        return distance
    
    @staticmethod
    def calculate_sway(com_trajectory):
        """
        Calculate body sway from CoM trajectory over time.
        
        Args:
            com_trajectory: Array of CoM positions over time (N x 3)
            
        Returns:
            dict: Sway metrics (total, mediolateral, anteroposterior)
        """
        if len(com_trajectory) < 2:
            return None
        
        # Calculate displacements
        displacements = np.diff(com_trajectory, axis=0)
        
        # Total path length
        total_sway = np.sum(np.linalg.norm(displacements, axis=1))
        
        # Mediolateral sway (X direction)
        ml_sway = np.sum(np.abs(displacements[:, 0]))
        
        # Anteroposterior sway (Z direction)
        ap_sway = np.sum(np.abs(displacements[:, 2]))
        
        # Vertical sway
        vertical_sway = np.sum(np.abs(displacements[:, 1]))
        
        return {
            'total': total_sway,
            'mediolateral': ml_sway,
            'anteroposterior': ap_sway,
            'vertical': vertical_sway
        }


class PhysicalIntelligence:
    """
    Complete physical intelligence analysis.
    """
    
    def __init__(self, body_mass=70.0):
        """
        Initialize physical intelligence analyzer.
        
        Args:
            body_mass: Total body mass in kg
        """
        self.com_calculator = CenterOfMassCalculator(body_mass)
        self.balance_analyzer = BalanceAnalyzer()
    
    def analyze_frame(self, joint_positions):
        """
        Complete physical analysis for a single frame.
        
        Args:
            joint_positions: Dict of {joint_name: [x, y, z]}
            
        Returns:
            dict: Physical intelligence metrics
        """
        results = {}
        
        # Center of mass
        com = self.com_calculator.calculate_com_relative_to_hip(joint_positions)
        results['com'] = com
        
        if com is not None:
            results['com_height'] = com[1]  # Y coordinate (height above hips)
        
        # Base of support
        if 'LEFT_ANKLE' in joint_positions and 'RIGHT_ANKLE' in joint_positions:
            bos = self.balance_analyzer.calculate_base_of_support(
                joint_positions['LEFT_ANKLE'],
                joint_positions['RIGHT_ANKLE']
            )
            results['base_of_support'] = bos
            
            if com is not None:
                # Balance metric
                stability = self.balance_analyzer.calculate_com_to_bos_distance(
                    com, bos
                )
                results['stability_margin'] = stability
        
        return results
    
    def analyze_motion(self, joint_positions_sequence):
        """
        Analyze motion over time.
        
        Args:
            joint_positions_sequence: List of joint position dicts
            
        Returns:
            dict: Motion analysis including CoM trajectory and sway
        """
        com_trajectory = []
        
        for joint_positions in joint_positions_sequence:
            com = self.com_calculator.calculate_com_relative_to_hip(joint_positions)
            if com is not None:
                com_trajectory.append(com)
        
        if not com_trajectory:
            return None
        
        com_trajectory = np.array(com_trajectory)
        
        results = {
            'com_trajectory': com_trajectory,
            'com_mean': np.mean(com_trajectory, axis=0),
            'com_std': np.std(com_trajectory, axis=0),
        }
        
        # Calculate sway
        sway = self.balance_analyzer.calculate_sway(com_trajectory)
        results['sway'] = sway
        
        return results


def test_physics_intelligence():
    """
    Test physical intelligence calculations.
    """
    print("[TEST] Physical Intelligence Layer")
    
    # Create synthetic joint positions (standing pose)
    joint_positions = {
        'MID_HIP': np.array([0.0, 0.0, 0.0]),
        'LEFT_HIP': np.array([-0.15, 0.0, 0.0]),
        'RIGHT_HIP': np.array([0.15, 0.0, 0.0]),
        'LEFT_SHOULDER': np.array([-0.20, -0.40, 0.0]),
        'RIGHT_SHOULDER': np.array([0.20, -0.40, 0.0]),
        'LEFT_ELBOW': np.array([-0.20, -0.15, 0.0]),
        'RIGHT_ELBOW': np.array([0.20, -0.15, 0.0]),
        'LEFT_WRIST': np.array([-0.20, 0.05, 0.0]),
        'RIGHT_WRIST': np.array([0.20, 0.05, 0.0]),
        'LEFT_KNEE': np.array([-0.15, 0.45, 0.0]),
        'RIGHT_KNEE': np.array([0.15, 0.45, 0.0]),
        'LEFT_ANKLE': np.array([-0.15, 0.90, 0.0]),
        'RIGHT_ANKLE': np.array([0.15, 0.90, 0.0]),
    }
    
    # Test CoM calculation
    com_calc = CenterOfMassCalculator(total_body_mass=70.0)
    com = com_calc.calculate_com_relative_to_hip(joint_positions)
    
    print(f"\nCenter of Mass: {com}")
    print(f"  X (lateral): {com[0]:.4f}")
    print(f"  Y (vertical): {com[1]:.4f}")
    print(f"  Z (anteroposterior): {com[2]:.4f}")
    
    # Test balance analysis
    pi = PhysicalIntelligence(body_mass=70.0)
    frame_analysis = pi.analyze_frame(joint_positions)
    
    print(f"\nBalance Analysis:")
    if 'stability_margin' in frame_analysis:
        print(f"  Stability margin: {frame_analysis['stability_margin']:.4f}")
    
    # Test motion analysis with multiple frames
    print("\n[TEST] Motion Analysis:")
    
    # Simulate 10 frames with slight movement
    sequence = []
    for i in range(10):
        joints = joint_positions.copy()
        # Add small random perturbations
        for key in joints:
            joints[key] = joints[key] + np.random.normal(0, 0.01, 3)
        sequence.append(joints)
    
    motion_analysis = pi.analyze_motion(sequence)
    
    if motion_analysis:
        print(f"  CoM mean position: {motion_analysis['com_mean']}")
        print(f"  CoM std deviation: {motion_analysis['com_std']}")
        if 'sway' in motion_analysis:
            sway = motion_analysis['sway']
            print(f"  Total sway: {sway['total']:.4f}")
            print(f"  Mediolateral sway: {sway['mediolateral']:.4f}")
            print(f"  Anteroposterior sway: {sway['anteroposterior']:.4f}")
    
    print("\n[TEST COMPLETE] Physical intelligence tests passed")


if __name__ == "__main__":
    test_physics_intelligence()