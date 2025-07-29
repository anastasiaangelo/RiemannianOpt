import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import configparser
import ast
from dotenv import load_dotenv
from supporting_functions import get_hyperparameters, log_info
import simulations_production as SP
import traceback


load_dotenv()

class Simulation:
    def __init__(self, match, simulation_name):
        print(f"Initializing Simulation with simulation_name: {repr(simulation_name)}")
        self.simulation_name = simulation_name.strip()

        # Load configuration
        config_path = "/home/b/aag/ropt-aqc/production/config.ini"
        self.config = configparser.ConfigParser()
        self.config.optionxform = str  # Preserve case sensitivity
        self.config.read(config_path)

        # Normalize section names to avoid whitespace mismatches
        normalized_sections = {s.strip(): s for s in self.config.sections()}
        print(f"Normalized Sections: {list(normalized_sections.keys())}")  # Debugging

        if self.simulation_name not in normalized_sections:
            raise ValueError(
                f"Error: Section '{self.simulation_name}' not found in config file.\n"
                f"Available Sections: {list(normalized_sections.keys())}"
            )

        self.simulation_name = normalized_sections[self.simulation_name]

        # Extract parameters safely
        try:
            self.params = {k: ast.literal_eval(v) for k, v in self.config[self.simulation_name].items()}
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Error parsing values in section '{self.simulation_name}': {e}")

        self.match = match
        self.hyperparameters = get_hyperparameters(self.match, *self.params.values())

        print(f"Simulation Parameters: {self.params}")  # Debugging
        print(f"Initialized Simulation with match: {match}, simulation_name: {self.simulation_name}")


    def PXP_AQC_forward_compression_sweep(self):
        num_sites, final_time, num_steps = self.hyperparameters
        log_info(f"Running PXP_AQC_forward_compression_sweep with num_sites: {num_sites}, final_time: {final_time}, num_steps: {num_steps}")
        SP.PXP_AQC_forward_compression(num_sites, final_time, num_steps)

    def PXP_AQC_reverse_compression_sweep(self):
        num_sites, final_time, num_steps = self.hyperparameters
        log_info(f"Running PXP_AQC_reverse_compression_sweep with num_sites: {num_sites}, final_time: {final_time}, num_steps: {num_steps}")
        SP.PXP_AQC_reverse_compression(num_sites, final_time, num_steps)

    def PXP_hybrid_forward_compression_sweep(self):
        num_sites, num_steps, final_time = self.hyperparameters
        log_info(f"Running PXP_hybrid_forward_compression_sweep with num_sites: {num_sites}, final_time: {final_time}, num_steps: {num_steps}")
        SP.PXP_hybrid_forward_compression(num_sites, num_steps, final_time)
    
    def PXP_hybrid_reverse_compression_sweep(self):
        num_sites, num_steps, final_time = self.hyperparameters
        log_info(f"Running PXP_hybrid_reverse_compression_sweep with num_sites: {num_sites}, final_time: {final_time}, num_steps: {num_steps}")
        SP.PXP_hybrid_reverse_compression(num_sites, num_steps, final_time)

    

if __name__ == "__main__":
    log_path = os.path.join(os.getcwd(), 'log.txt')
    
    git_commit = sys.argv[1]
    match = int(sys.argv[2])
    sweep_type = sys.argv[3]
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))  # Get Slurm Job Array Index

    log_info(f"Git commit: {git_commit}")

    simulation_name = "PXP_compression_sweep"

    # Read the config file
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity
    config.read("/home/b/aag/ropt-aqc/production/config.ini")

    # print("CONFIG SECTIONS FOUND:", config.sections())
    # print("SIMULATION NAME:", repr(simulation_name))

    param_list = []
    normalized_sections = {s.strip(): s for s in config.sections()}
    if simulation_name not in normalized_sections:
        raise ValueError(f"Section '{simulation_name}' not found in config.")

    section = normalized_sections[simulation_name]

    for num_site in ast.literal_eval(config[section]["num_sites_arr"]):
        for num_step in ast.literal_eval(config[section]["num_steps_arr"]):
            for final_time in ast.literal_eval(config[section]["final_time_arr"]):
                param_list.append((num_site, num_step, final_time))

    if task_id >= len(param_list):
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID {task_id}, max {len(param_list) - 1}")

    selected_params = param_list[task_id]
    print(f"Running simulation {task_id} with parameters: {selected_params}")

    # Initialize the simulation object with the selected parameters
    simulation = Simulation(match, simulation_name)
    try:
            method_map = {
                "AQC_forward": simulation.PXP_AQC_forward_compression_sweep,
                "AQC_reverse": simulation.PXP_AQC_reverse_compression_sweep,
                "hybrid_forward": simulation.PXP_hybrid_forward_compression_sweep,
                "hybrid_reverse": simulation.PXP_hybrid_reverse_compression_sweep,
            }

            if sweep_type not in method_map:
                raise ValueError(f"Unknown sweep type: {sweep_type}")

            with open(f"{sweep_type}.out", "w") as f_out, \
                open(f"{sweep_type}.err", "w") as f_err:
                try:
                    sys.stdout = f_out
                    sys.stderr = f_err
                    method_map[sweep_type]()
                finally:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__

            log_info(f"Successfully completed sweep: {sweep_type}")

    except Exception as e:
        print(f"Error occurred during sweep '{sweep_type}':")
        traceback.print_exc()
        log_info(f"Unexpected error: {e}")
