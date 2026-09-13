"""
Shows exactly what RDAP and WHOIS each return for a domain, with no
processing applied — so we can see precisely which one is giving bad
data instead of guessing.

Usage: python3 debug_domain_age.py airbnb.com
"""

import sys
import requests
from datetime import datetime

try:
    import whois
except ImportError:
    whois = None


def debug_rdap(domain, timeout=6):
    print(f"--- RDAP for {domain} ---")
    try:
        response = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout)
        response.raise_for_status()
        data = response.json()

        events = data.get('events', [])
        if not events:
            print("No 'events' field in the RDAP response at all.")
        for event in events:
            print(f"  {event.get('eventAction'):20} {event.get('eventDate')}")
    except Exception as e:
        print(f"RDAP failed: {e}")
    print()


def debug_whois(domain):
    print(f"--- WHOIS for {domain} ---")
    if whois is None:
        print("python-whois not installed.")
        return
    try:
        w = whois.whois(domain)
        print("creation_date raw:", repr(w.creation_date))
        print("updated_date raw:", repr(w.updated_date))
    except Exception as e:
        print(f"WHOIS failed: {e}")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 debug_domain_age.py airbnb.com")
        sys.exit(1)

    domain = sys.argv[1]
    debug_rdap(domain)
    debug_whois(domain)