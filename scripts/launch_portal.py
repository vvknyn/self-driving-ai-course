#!/usr/bin/env python3
"""
Launch Portal: Serves the interactive visual explainers locally.
Usage:
    python scripts/launch_portal.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean terminal output
        if "200 -" in args[0] or "304 -" in args[0]:
            return
        super().log_message(format, *args)

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    
    url = f"http://localhost:{PORT}/visual_explainers/index.html"
    print("\n" + "="*70)
    print("🚀 MINI-FSD VISUAL EXPLAINER PORTAL")
    print("="*70)
    print(f"Serving repository at: {repo_root}")
    print(f"Interactive Portal URL: {url}")
    print("Press Ctrl+C to stop the server.")
    print("="*70 + "\n")
    
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
