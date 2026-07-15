#!/bin/bash

# Test internet connectivity with several protocols and tools.

# Ping a reliable IP first because it avoids DNS resolution.
if ping -c 2 -W 2 8.8.8.8 &>/dev/null; then
    echo "Internet reachable (ping to 8.8.8.8)"
    exit 0
fi

# Check DNS resolution and ICMP together.
if ping -c 2 -W 2 google.com &>/dev/null; then
    echo "Internet reachable (ping to google.com)"
    exit 0
fi

# HTTP may still work when ICMP is blocked.
if curl -fs --max-time 5 https://www.google.com >/dev/null; then
    echo "Internet reachable (HTTPS with curl)"
    exit 0
fi

if wget -q --spider --timeout=5 https://www.google.com 2>/dev/null; then
    echo "Internet reachable (HTTPS with wget)"
    exit 0
fi

echo "No internet connection detected"
exit 1
