import argparse
import os
from multiprocessing import Pool
from numpy.core.numeric import normalize_axis_tuple
import psutil
import datetime as dt
import numpy as np
from dateutil.parser import parse
import pyDOE


def run_in_parallel(num_simulations, num_parallel, config, log_folder, verbose,
                    book_freq, hist_date, mkt_start_time, mkt_end_time, noise):

    global_seeds = np.random.randint(0, 2 ** 31, num_simulations)
    print(f'Global Seeds: {global_seeds}')
    
    # create configerations to simulate
    agent_config = np.empty((num_simulations, 4))
    count = 0

    # loop until correct number of valid samples are obtained
    while count < num_simulations:
        
        # sample an array of 4 random integers between 0 and 999 inclusive
        combination = [np.random.randint(0, 1000),
                        np.random.randint(0, 200),
                        np.random.randint(0, 200),
                        np.random.randint(0, 200)]
        
        # total number of agents has to be 999 (+1 later for the market maker)
        if sum(combination) == 999:
            
            # assign valid combination to a list of configerations
            agent_config[count] = combination
            count += 1
            
            # ZIP agent is computationally expensive, so only upto 200
            if combination[1] < 200:
                
                # Too many technical agents lead to the system diverging
                if combination[2] < 100:
                    
                    # Too many technical agents lead to the system diverging
                    if combination[3] < 100:
                        
                        # assign valid combination to a list of configerations
                        agent_config[count] = combination
                        count += 1
                        

    # initialise processes
    processes = []

    # iterate over number of simulations
    for i in range(num_simulations):

        # set seed according to global seed
        seed = global_seeds[i]

        # assign number of atents according to predetermined configeration
        zi_count = int(agent_config[i][0])
        zip_count = int(agent_config[i][1])
        mmt_count = int(agent_config[i][2])
        mr_count = int(agent_config[i][3])
        mm_count = 1

        print(f"current config: {zi_count}, {zip_count}, {mmt_count}, {mr_count}")


        processes.append(f'python -u abides.py -c {config} -l {log_folder}_config_{zi_count}_{zip_count}_{mmt_count}_{mr_count}_{mm_count} \
                        {"-v" if verbose else ""} -s {seed} -b {book_freq} -d {hist_date} -st {mkt_start_time} -et {mkt_end_time} \
                        -zi {zi_count} -zip {zip_count} -mmt {mmt_count} -mr {mr_count} -mm {mm_count} -n {noise}')

    print(processes)  
    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')
    parser.add_argument('-c', '--config', 
                        required=True,
                        help='Name of config file to execute'
                        )
    parser.add_argument('-ns', '--num_simulations', 
                        type=int,
                        default=1,
                        help='Total number of simulations to run')
    parser.add_argument('-np', '--num_parallel', 
                        type=int,
                        default=None,
                        help='Number of simulations to run in parallel')
    parser.add_argument('-l', '--log_folder',
                        required=True,
                        help='Log directory name')
    parser.add_argument('-b', '--book_freq', 
                        default=None,
                        help='Frequency at which to archive order book for visualization'
                        )
    parser.add_argument('-n', '--obs_noise', 
                        type=float, 
                        default=1000000,
                        help='Observation noise variance for zero intelligence agents (sigma^2_n)'
                        )
    parser.add_argument('-o', '--log_orders',
                        action='store_true',
                        help='Log every order-related action by every agent.'
                        )
    parser.add_argument('-s', '--seed', 
                        type=int, 
                        default=None,
                        help='numpy.random.seed() for simulation'
                        )
    parser.add_argument('-v', '--verbose', 
                        action='store_true',
                        help='Maximum verbosity!'
                        )
    parser.add_argument('--config_help', 
                        action='store_true',
                        help='Print argument options for this config file'
                        )
    parser.add_argument('-d', '--historical_date',
                        required=True,
                        help='historical date being simulated in format YYYYMMDD.'
                        )
    parser.add_argument('-st', '--start_time',
                        default='09:30:00',
                        help='Starting time of simulation.'
                        )
    parser.add_argument('-et', '--end_time',
                        default='16:00:00',
                        help='Ending time of simulation.'
                        )

    args, remaining_args = parser.parse_known_args()

    seed            = args.seed
    num_simulations = args.num_simulations
    num_parallel    = args.num_parallel if args.num_parallel else psutil.cpu_count() # count of the CPUs on the machine
    config          = args.config
    log_folder      = args.log_folder
    verbose         = args.verbose
    book_freq       = args.book_freq
    hist_date       = args.historical_date
    mkt_start_time  = args.start_time
    mkt_end_time    = args.end_time
    noise           = args.obs_noise


    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Configuration: {config}')

    np.random.seed(seed)

    run_in_parallel(num_simulations = num_simulations,
                    num_parallel    = num_parallel,
                    config          = config,
                    log_folder      = log_folder,
                    verbose         = verbose,
                    book_freq       = book_freq,
                    hist_date       = hist_date,
                    mkt_start_time  = mkt_start_time,
                    mkt_end_time    = mkt_end_time,
                    noise = noise)

    end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {end_time - start_time}')
