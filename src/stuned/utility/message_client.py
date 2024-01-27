import json
import os
import select
import socket
import time


class MessageType:
    JOB_STARTED = 0
    JOB_ERROR = 1
    JOB_RESULT_UPDATE = 2
    JOB_FINISHED = 3


class MessageClient:
    def __init__(self, server_ip, server_port, logger):
        self.server_ip = server_ip
        self.server_port = server_port
        self.logger = logger
        self.socket = None
        self.connect()

        self.message_queue = []

        self.job_id = self.get_job_name()

        # store for backup - if we can't connect with the server, we can still save the results locally
        # later, we can sync the results with the server
        self.local_csv_info = {}
        self.could_connect = True

    def connect(self, force_reconnect=False):
        self.logger.log(f"Attempting to connect to the server... force_reconnect={force_reconnect}")

        if self.socket and not force_reconnect:
            self.logger.log("Already connected.")
            self.could_connect = True
            return

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.logger.log("Connection established successfully.")
        except Exception as e:
            self.logger.log(f"Error trying to open the socket: {e}")
            self.socket = None
        if self.socket is not None:
            self.could_connect = True

    def get_job_name(self):
        return int(os.environ.get("SLURM_JOB_ID", str(os.getpid())))

    def send_start_command(self):
        # job name is either SLURM job id or "local" PID
        self.send_message(MessageType.JOB_STARTED, None, None, sync=True)

    def send_key_val(self, key, val, sync=False):
        self.send_message(MessageType.JOB_RESULT_UPDATE, key, val, sync=sync)

    def sync_with_remote(self):
        if not self.socket:
            self.connect()
        if len(self.message_queue) == 0:
            return

        retries = 10
        retry_delay = 2

        message_str = ""

        if not self.could_connect:
            # We tried to connect before but couldn't -- fall back to local dict writing. We can sync later.
            pass
        else:
            for _ in range(retries):
                try:
                    all_messages = []
                    for message_type, message_key, message_value in self.message_queue:
                        # Serialize the message as JSON
                        message_data = {
                            "type": message_type,
                            "key": str(message_key),
                            "value": str(message_value),
                        }
                        all_messages.append(message_data)

                    message_str = json.dumps(
                        {"job_id": self.job_id, "messages": all_messages}
                    ).encode("utf-8")

                    # Send the length of the message first
                    # Send the length of the message first
                    message_length = len(message_str)
                    self.socket.setblocking(0)  # Set socket to non-blocking

                    # Wait for socket to be ready for sending data
                    ready_to_send = select.select([self.socket], [], [], 10)
                    if ready_to_send[0]:
                        self.socket.sendall(message_length.to_bytes(4, "big"))
                        self.socket.sendall(message_str)
                    else:
                        self.logger.log("Timeout occurred while sending data.")
                        return

                    # Wait for socket to be ready for receiving data
                    ready_to_receive = select.select([self.socket], [], [], 10)
                    if ready_to_receive[0]:
                        data = self.socket.recv(1024)
                        if data == b"ACK":
                            self.logger.log("Messages sent successfully!")
                            self.message_queue = []  # Clear the queue after successful send
                            return
                        else:
                            self.logger.log("No ACK received. Retrying...")
                    else:
                        self.logger.log("Timeout occurred while waiting for ACK.")
                        return
                    # message_length = len(message_str)
                    # self.socket.sendall(message_length.to_bytes(4, "big"))
                    #
                    # # Send the actual message
                    # self.socket.sendall(message_str)
                    #
                    # data = self.socket.recv(1024)
                    # if data == b"ACK":
                    #     self.logger.log("Messages sent successfully!")
                    #     self.message_queue = []  # Clear the queue after successful send
                    #     return
                    # else:
                    #     self.logger.log("No ACK received. Retrying...")
                except Exception as e:
                    self.logger.log(f"Error: {e}. Retrying in {retry_delay} seconds...")
                    self.logger.log(f"Attempted to log: \n {message_str}")
                    print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # logger.log("Reconnecting...")
                    self.connect(force_reconnect=True)  # Try to reconnect

        self.logger.log("Failed to send messages after multiple retries.")

    def send_message(self, message_type, message_key, message_value, sync=False):
        # add to local thingie if message_type is JOB_RESULT_UPDATE
        if message_type == MessageType.JOB_RESULT_UPDATE:
            self.local_csv_info[message_key] = message_value
        self.message_queue.append((message_type, message_key, message_value))
        if sync:
            self.sync_with_remote()
