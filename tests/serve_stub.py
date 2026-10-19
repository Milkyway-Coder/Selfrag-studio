"""Runs the real Flask app with stubbed embedder/LLM for UI testing: python tests/serve_stub.py [port]"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "stubs")); sys.path.insert(0, ROOT)
os.chdir(tempfile.mkdtemp()); os.environ["GROQ_API_KEY"] = "x"
import app as A
A.app.run(port=int(sys.argv[1]) if len(sys.argv) > 1 else 5055, threaded=True, use_reloader=False)
