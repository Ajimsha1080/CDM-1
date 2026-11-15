#!/usr/bin/env python3
"""CDM-1 Developer Playground Unified Launcher.
Launches the FastAPI backend and Next.js frontend with one command,
performing prerequisite checks and opening the browser automatically.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
import signal
from typing import List, Optional

BACKEND_PORT = 8000
FRONTEND_PORT = 3000
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAYGROUND_DIR = os.path.join(ROOT_DIR, "playground")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def check_prerequisites():
    print("=" * 65)
    print("  CDM-1 DEVELOPER PLAYGROUND LAUNCHER")
    print("=" * 65)
    print(f"[*] Root directory: {ROOT_DIR}")
    print(f"[*] Python executable: {sys.executable} ({sys.version.split()[0]})")

    # Check Node / npm
    try:
        node_ver = subprocess.check_output(["node", "--version"], text=True, stderr=subprocess.DEVNULL).strip()
        npm_ver = subprocess.check_output(["npm", "--version"], shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        print(f"[*] Node.js: {node_ver} | npm: {npm_ver}")
    except Exception as e:
        print("[!] ERROR: Node.js and npm are required to run the Developer Playground.")
        print("    Please install Node.js (v18+) from https://nodejs.org/")
        sys.exit(1)

    # Check node_modules in playground
    nm_dir = os.path.join(PLAYGROUND_DIR, "node_modules")
    if not os.path.exists(nm_dir):
        print("[*] Installing playground frontend dependencies (npm install)...")
        res = subprocess.run(["npm", "install"], shell=True, cwd=PLAYGROUND_DIR)
        if res.returncode != 0:
            print("[!] ERROR: Failed to install frontend dependencies.")
            sys.exit(1)
        print("[+] Frontend dependencies installed successfully.")


def main():
    check_prerequisites()

    processes: List[subprocess.Popen] = []

    def cleanup(*args):
        print("\n[*] Shutting down CDM-1 Developer Playground...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[+] All processes stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    # 1. Start Backend if not already running on port 8000
    if is_port_in_use(BACKEND_PORT):
        print(f"[!] Port {BACKEND_PORT} is already in use. Assuming CDM-1 backend is running.")
    else:
        print(f"[*] Launching CDM-1 FastAPI backend on port {BACKEND_PORT}...")
        backend_env = os.environ.copy()
        backend_env["PYTHONPATH"] = ROOT_DIR
        backend_proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT_DIR, "api", "main.py")],
            cwd=ROOT_DIR,
            env=backend_env,
        )
        processes.append(backend_proc)

        # Wait for backend to be ready
        print("[*] Waiting for backend readiness probe...")
        ready = False
        import urllib.request
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{BACKEND_PORT}/v1/health", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(1)

        if ready:
            print(f"[+] CDM-1 REST API is active at http://127.0.0.1:{BACKEND_PORT}")
        else:
            print("[!] Warning: Backend probe timed out; proceeding to launch frontend...")

    # 2. Start Playground Frontend
    if is_port_in_use(FRONTEND_PORT):
        print(f"[!] Port {FRONTEND_PORT} is already in use. Assuming playground is running.")
    else:
        print(f"[*] Launching Next.js Playground on port {FRONTEND_PORT}...")
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            shell=True,
            cwd=PLAYGROUND_DIR,
        )
        processes.append(frontend_proc)

    # 3. Open Browser
    playground_url = f"http://localhost:{FRONTEND_PORT}"
    print(f"\n[+] Developer Playground is available at: {playground_url}")
    print("[*] Press Ctrl+C at any time to stop the server.\n")

    time.sleep(2)
    try:
        webbrowser.open(playground_url)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    print(f"[!] Process {p.pid} terminated unexpectedly with code {p.returncode}.")
                    cleanup()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
