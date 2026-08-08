#!/usr/bin/env bash
# entrypoint.sh — start SSH + the FLUX server, log everything.
# Exit-1 crash-proofing: if the server dies, keep SSH alive and log so we can diagnose.
set -x
echo "[ENTRYPOINT] Starting at $(date)" | tee /tmp/flux_debug.log

# 1. Start SSH server (so we can exec in and read logs/stdout)
echo "[ENTRYPOINT] Starting SSH..." | tee -a /tmp/flux_debug.log
mkdir -p /run/sshd
# root login with the injected public key (PUBLIC_KEY env) or password root/root
if [ -n "$PUBLIC_KEY" ]; then
  mkdir -p /root/.ssh && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
echo 'root:root' | chpasswd 2>/dev/null || true
/usr/sbin/sshd 2>&1 | tee -a /tmp/flux_debug.log || /usr/sbin/sshd -D 2>&1 | tee -a /tmp/flux_debug.log &
echo "[ENTRYPOINT] SSH pid started" | tee -a /tmp/flux_debug.log

# 2. Start the FLUX server, capturing logs to file + console (tee). Run in foreground so
#    the container stays alive; a crash logs and we keep running (loop-safe).
echo "[ENTRYPOINT] Starting FLUX server..." | tee -a /tmp/flux_debug.log
while true; do
  echo "[ENTRYPOINT] FLUX attempt at $(date)" | tee -a /tmp/flux_debug.log
  cd /app
  python3 -u /app/flux_http_server.py >> /tmp/flux_debug.log 2>&1
  echo "[ENTRYPOINT] FLUX exited code=$? at $(date) — restarting in 5s" | tee -a /tmp/flux_debug.log
  sleep 5
done
