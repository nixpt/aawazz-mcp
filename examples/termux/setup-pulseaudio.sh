#!/usr/bin/env bash
# Start the PulseAudio daemon on the Termux side so paplay (and
# `speak(play=true)`) can reach Android's audio out.
#
# Run this in a **Termux** shell, NOT inside proot-distro. The daemon
# binds 127.0.0.1; aawazz-mcp inside proot reaches it via PULSE_SERVER.
#
# Inside proot, export PULSE_SERVER=127.0.0.1 (most proot-distro setups
# already do). Verify the bridge with `pactl info` from inside proot:
# you should see "Server String: 127.0.0.1" and a non-zero protocol
# version, not "Connection refused".

set -euo pipefail

command -v pulseaudio >/dev/null || pkg install -y pulseaudio

# --exit-idle-time=-1 keeps the daemon alive even when no client is
# connected; otherwise it self-shuts after ~20 s of idle and the next
# speak() call hits "Connection refused" again.
pulseaudio --start \
    --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1" \
    --exit-idle-time=-1

pactl info | head -5
