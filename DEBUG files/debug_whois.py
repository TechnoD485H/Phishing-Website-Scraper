"""
Prints the RAW WHOIS response for a domain, with no processing applied —
useful for figuring out exactly what our code is (mis)interpreting.
"""

import sys
import whois

def debug_whois(domain):
    print(f"Querying WHOIS for: {domain}\n")
    w = whois.whois(domain)

    print("creation_date raw value:", repr(w.creation_date))
    print("updated_date raw value:", repr(w.updated_date))
    print("expiration_date raw value:", repr(w.expiration_date))
    print()
    print("Full record:")
    print(w)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 debug_whois.py airbnb.com")
        sys.exit(1)
    debug_whois(sys.argv[1])