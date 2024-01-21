import requests


class CLUSTER:
    OWL1 = "owl1"
    GAL = "gal"
    UNKNOWN = "unknown"


def get_current_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        response.raise_for_status()
        return response.json()["origin"]
    except requests.RequestException as e:
        print(f"Error fetching IP address: {e}")
        return None


def determine_region(current_ip, region_owl1_ips, region_gal_ips):
    if current_ip in region_owl1_ips:
        return CLUSTER.OWL1
    elif current_ip in region_gal_ips:
        return CLUSTER.GAL
    else:
        return CLUSTER.UNKNOWN


def get_region():
    region_owl1_ips = ["134.2.168.52", "134.2.168.72"]  # IPs for r1
    region_gal_ips = ["134.2.168.43"]  # IPs for gal

    my_ip = get_current_ip()
    if my_ip:
        region = determine_region(my_ip, region_owl1_ips, region_gal_ips)
        return region
    return CLUSTER.UNKNOWN
