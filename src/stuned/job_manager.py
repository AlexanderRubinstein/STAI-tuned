"""
TODO: make this functional
Job manager that manages jobs and keeps track of their status.
Ideally works for both local and remote runs.
"""
import json
from multiprocessing import Manager, Process
import socket
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from time import sleep
import time
from typing import List

from stuned.job import Job, find_job_idx

def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class JobManager():
    def __init__(self, local_run, open_socket, logger):
        self.local_run = local_run
        self.logger = logger
        
        self.server_ip = None
        self.server_port = None
        
        self.manager = Manager()
        self.manager_lock = self.manager.Lock()
        
        self.jobs : List[Job] = self.manager.list()  # Shared list
        self.open_socket = open_socket
        
        if open_socket:
            self.server_info = self.manager.dict()  # Shared dictionary
            self.queue = self.manager.Queue()  # Shared queue
            self.server_process = Process(target=self.run_server)
            self.server_process.start()

            # Wait until server IP and port are set
            while 'ip' not in self.server_info or 'port' not in self.server_info:
                sleep(1)

            self.server_ip = self.server_info['ip']
            self.server_port = self.server_info['port']
    
    def process_job_message(self, message):
        """
        Process a message from a job.
        
        args:
            message (dict): message from a job in JSON format. 
        
        """
        assert "job_id" in message, "Job ID not found in message"
        job_id = message["job_id"]
        
        with self.manager_lock:
            job_idx = find_job_idx(self.jobs, job_id)
            if job_idx is not None:
                job = self.jobs[job_idx]
            else:
                self.logger.log(f"Job {job_id} not found")
                
                # Should we create a new job?
                job = Job(job_id, None, None, None)
                self.jobs.append(job)
                
            # Unsure if this will update the job in the list
            job.process_message(message)
            
            self.jobs[job_idx] = job
                
        

    def handle_client(self, client, queue):
        with client:
            while True:
                # Read the first 4 bytes to get the message length
                length_data = client.recv(4)
                if not length_data:
                    break

                message_length = int.from_bytes(length_data, 'big')

                # Read the actual message
                data = client.recv(message_length)
                if not data:
                    self.logger.log('Warning: Connection closed by client')
                    break

                # Deserialize the message
                try:
                    message_obj = json.loads(data.decode('utf-8'))
                except:
                    self.logger.log('Warning: Failed to deserialize message')
                    self.logger.log(f"Tried to deserialize: {data}\n which decoded to {data.decode('utf-8')}")
                    break
                assert "job_id" in message_obj, "Job ID not found in message"
                assert "messages" in message_obj, "Messages not found in message"
                
                # Might need to update or add a new job
                self.process_job_message(message_obj)

                # self.logger.log(f"Received a message from {message_obj['job_id']}")
                queue.put(message_obj)
                client.sendall(b'ACK')

                time.sleep(0.01)  # Sleep for 10 milliseconds


    def run_server(self):
        HOST = get_my_ip()  # Assuming you have this function from the previous example
        PORT = 0  # Let OS choose a free port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            actual_port = s.getsockname()[1]  # Get the port chosen by the OS

            self.server_info['ip'] = HOST
            self.server_info['port'] = actual_port
            
            self.logger.log(f'Server listening on {HOST}:{actual_port}')

            while True:
                client, addr = s.accept()
                self.logger.log('Connected by', addr)
                # Using threading here to handle multiple clients in the same process
                threading.Thread(target=self.handle_client, args=(client, self.queue)).start()

            
    def process_messages(self):
        while True:
            if not self.queue.empty():
                message = self.queue.get()
                # Process the message here
                self.logger.log(f"Processing: {message}")

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            
    @property
    def get_running_jobs_count(self):
        pass

    @property
    def get_finished_jobs_count(self):
        pass

    @property
    def get_failed_jobs_count(self):
        pass

