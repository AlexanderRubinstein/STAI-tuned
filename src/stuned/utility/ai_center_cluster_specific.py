import os
import requests

class CLUSTER:
    FERRANTI = "ferranti"
    GALVANI = "galvani"
    UNKNOWN = "unknown"

# TODO: probably a better way to do this exists :)
def get_current_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        response.raise_for_status()
        return response.json()["origin"]
    except requests.RequestException as e:
        print(f"Error fetching IP address: {e}")
        return None


def determine_cluster_by_path():
    # Check for the presence of specific directories to identify the cluster
    if os.path.exists("/weka"):
        return CLUSTER.FERRANTI
    elif os.path.exists("/mnt/lustre"):
        return CLUSTER.GALVANI
    return CLUSTER.UNKNOWN



def get_region():
    # region_owl1_ips = ["134.2.168.52", "134.2.168.72"]  # IPs for r1
    # region_gal_ips = ["134.2.168.43"]  # IPs for gal

    region = determine_cluster_by_path()
    # my_ip = get_current_ip()
    # if my_ip:
        # return region
    return region
