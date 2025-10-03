"""Utilities to launch the Streamlit portfolio with a temporary public URL."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress

from pyngrok import conf, ngrok


def _launch_streamlit(port: int) -> subprocess.Popen:
    """Start the Streamlit app on the requested port and return the process."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
    ]
    return subprocess.Popen(command)


def main() -> None:
    """Open a public ngrok tunnel and run the Streamlit application."""
    port = int(os.environ.get("STREAMLIT_PORT", "8501"))
    region = os.environ.get("NGROK_REGION")

    if region:
        conf.get_default().region = region

    public_tunnel = ngrok.connect(port, proto="http")
    print("Streamlit public URL:", public_tunnel.public_url, flush=True)

    process = _launch_streamlit(port)

    try:
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        with suppress(ProcessLookupError):
            process.send_signal(signal.SIGINT)
        with suppress(ProcessLookupError):
            process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        ngrok.disconnect(public_tunnel.public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
