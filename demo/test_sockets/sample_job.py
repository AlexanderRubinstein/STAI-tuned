import random
import socket
import time

from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
from stuned.utility.logger import RedneckLogger
from stuned.utility.message_client import MessageClient, MessageType



def socket_experiment(experiment_config,
    logger : RedneckLogger,
    processes_to_kill_before_exiting):
    
    # hopefully the ip is contained within the experiment config
    # logger.log("sup zoomer")
    # logger.log(experiment_config["logging"]["server_ip"])
    # logger.log(experiment_config["logging"]["server_port"])
    # server_ip, server_port = experiment_config["logging"]["server_ip"], experiment_config["logging"]["server_port"]

    # time_init_msg_client = time.time()
    # msg_client = MessageClient(server_ip, server_port, logger)
    # time_end_msg_client = time.time()

    # msg_client.send_key_val("time_init_msg_client", time_end_msg_client - time_init_msg_client, sync=False)
    # import os
    # Get all env vars
    # env_vars = str(os.environ)
    # logger.log(env_vars)
    logger.socket_client.send_start_command()
    
    # msg_client.send_key_val("none", "Hello from mini job!", sync=True)
    
    all_cols = ["col1", "col2", "col3", "col4", "col5", "col6", "col7"]
    # either 0 or 1
    # make the seed random!  based on the timestamp u know
    random.seed(time.time())

    this_fails = random.randint(0, 1)

    logger.log(f"This fails? {this_fails}")
    wait_times_last = 0
    for i in range(1):
        # get some random data
        random_val = random.randint(0, 100)
        random_col = random.choice(all_cols)
        
        time_start = time.time()
        logger.socket_client.send_key_val(random_col, random_val, sync=True)
        time_end = time.time()
        logger.socket_client.send_key_val("time_send_msg", time_end - time_start, sync=False)
        time.sleep(10)

        if this_fails > 0:
            raise Exception("This job failed")
    
if __name__ == "__main__":
    prepare_wrapper_for_experiment()(
        socket_experiment
    )()

