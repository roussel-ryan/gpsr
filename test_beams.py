from distgen import Generator
import yaml

def generate_gaussian():
    input = """
        n_particle: 10000
        r_dist:
            sigma_xy:
                units: mm
                value: 1.0
            type: radial_gaussian
        z_dist:
          avg_z:
                units: mm
                value: 0
          sigma_z:
                units: mm
                value: 2.0
          type: gaussian
        
    """
