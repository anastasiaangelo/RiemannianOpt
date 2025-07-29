import configparser
import ast
import os
import sys
import logging
from dotenv import load_dotenv
from production.PXP_opt_parameter_sweep import PS
from itertools import product

load_dotenv()

def log_info(message):
    logging.info(message)


def get_hyperparameters(jobid, *args):
    """
    Generate all combinations of parameters and return the combination corresponding to the job ID.

    This function accepts a variable number of arguments. Each argument can be a list of values for a particular parameter or a single number. In the latter case, it will be treated as a list with one element.

    The function generates all combinations of these parameters using the Cartesian product. The combinations are zero-indexed, and the combination corresponding to the job ID is returned.

    """

    args = [[arg] if not isinstance(arg, list) else arg for arg in args]
    all_combinations = list(product(*args))

    if jobid <= len(all_combinations):
        return all_combinations[jobid - 1]
    else:
        raise ValueError("Job ID is out of range.")
    

class Simulation:
    def __init__(self, match, simulation_name):
        print(f"Initializing Simulation with simulation_name: {repr(simulation_name)}")
        self.simulation_name = simulation_name.strip()

        # Load configuration
        config_path = "/home/b/aag/ropt_aqc/production/config_sweep.ini"
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


    def run_PXP_hyperparameter_sweep(self):
        num_sites, num_steps, final_time, beta_1, beta_2, lr = self.hyperparameters
        log_info(f"Running noisy_xy_simulation_parameter_sweep with beta_1: {beta_1}, beta_2: {beta_2}, lr: {lr}")
        PS.PXP_hyperparameter_sweep(num_sites, num_steps, final_time, beta_1, beta_2, lr)
        
if __name__ == "__main__":
    log_path = os.path.join(os.getcwd(), 'log.txt')
    
    git_commit = sys.argv[1]
    match = int(sys.argv[2])
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))  # Get Slurm Job Array Index

    log_info(f"Git commit: {git_commit}")

    simulation_name = "run_PXP_hyperparameter_sweep"

    # Read the config file
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity
    config.read("/home/b/aag/proteinfolding/production/config.ini")

    param_list = []
    
    for num_sites in ast.literal_eval(config[simulation_name]["num_sites_arr"]):
        for num_steps in ast.literal_eval(config[simulation_name]["num_steps_arr"]):
            for final_time in ast.literal_eval(config[simulation_name]["final_time_arr"]):
                for beta_1 in ast.literal_eval(config[simulation_name]["beta_1_values"]):
                    for beta_2 in ast.literal_eval(config[simulation_name]["beta_2_values"]):
                        for lr in ast.literal_eval(config[simulation_name]["lr_values"]):
                            param_list.append((num_sites, num_steps, final_time, beta_1, beta_2, lr))

    if task_id >= len(param_list):
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID {task_id}, max {len(param_list) - 1}")

    selected_params = param_list[task_id]
    print(f"Running simulation {task_id} with parameters: {selected_params}")

    # Initialize the simulation object with the selected parameters
    simulation = Simulation(match, simulation_name)

    try:
        method = getattr(simulation, simulation_name)  # Get the method
        print(f"Calling method: {method}")  # Debugging
        method() 
        log_info(f"Successfully ran the simulation: {simulation_name}")

    except AttributeError as e:
        print(f"Error: The simulation '{simulation_name}' does not exist.")
        log_info(f"Error: The simulation '{simulation_name}' does not exist.")
        print(f"Actual Exception: {e}")  # Debugging - Print real exception

    except Exception as e:
        print(f"Unexpected error occurred while running simulation '{simulation_name}': {e}")
        log_info(f"Unexpected error: {e}")
