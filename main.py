import subprocess
import sys
import os

def main():
    script = os.path.join("scripts", "cardiac_report.py")
    cmd = [sys.executable, script] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
