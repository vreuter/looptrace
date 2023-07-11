"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import rouse_polymer
import sys
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SMC loop extrusion simulation and rouse polymer simulation.')
    parser.add_argument("--param_path", help="JSON parameters file path")
    parser.add_argument("--no_SMC_sim", help="Use if SMC positions pre-generated in simulation folder.", action='store_true')
    parser.add_argument("--ensemble", help='Use if simulating an ensemble of simulated loops rather than a dynamic simulation', action='store_true')
    parser.add_argument("--no_rouse_sim", help='If only running SMC simulation without polymer simulation.', action='store_true')
    args = parser.parse_args()
    with open(args.param_path, "r") as f:
        params = json.load(f)
    if not os.path.isdir(params['out_path']):
        os.makedirs(params['out_path'])
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = 0

    if args.no_SMC_sim:
        pass
    else:
        if args.ensemble:
            rouse_polymer.run_SMC_sim_random_halt(params, run_id = array_id)
        else:
            rouse_polymer.run_SMC_sim(params, run_id = array_id)
    
    if args.no_rouse_sim:
            pass
    else:
        rouse_polymer.run_rouse_SMC_sim(params, run_id = array_id)