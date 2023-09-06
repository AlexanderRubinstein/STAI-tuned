import random
import socket
import time

from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
from stuned.utility.message_client import MessageClient, MessageType



def socket_experiment(experiment_config,
    logger,
    processes_to_kill_before_exiting):
    
    # hopefully the ip is contained within the experiment config
    logger.log("sup zoomer")
    logger.log(experiment_config["logging"]["server_ip"])
    logger.log(experiment_config["logging"]["server_port"])
    server_ip, server_port = experiment_config["logging"]["server_ip"], experiment_config["logging"]["server_port"]
    
    msg_client = MessageClient(server_ip, server_port, logger)
    import os
    # Get all env vars
    env_vars = str(os.environ)
    logger.log(env_vars)
    msg_client.send_start_command()
    
    msg_client.send_key_val("none", "Hello from mini job!", sync=True)
    
    all_cols = ["col1", "col2", "col3", "col4", "col5", "col6", "col7"]
    wait_times_last = 0
    for i in range(100):
        # get some random data
        random_val = random.randint(0, 100)
        random_col = random.choice(all_cols)
        
        time_start = time.time()
        msg_client.send_key_val(random_col, random_val, sync=True)    
        time_end = time.time()
        msg_client.send_key_val("time_send_msg", time_end - time_start, sync=False)
        time.sleep(10)
    
if __name__ == "__main__":
    prepare_wrapper_for_experiment()(
        socket_experiment
    )()

