import json

class OSInspector:
    """
    Extract minimal OS-level information useful for agent planning.
    No hard-coded assumptions. No application scanning.
    Returns RAW info + a compact human/LLM-friendly summary.
    """

    def __init__(self, env):
        """
        Initialize `OSInspector` dependencies and runtime state.
        
        Parameters
        ----------
        env : Any
            Environment interface used to execute actions.
        
        Returns
        -------
        None
            No return value.
        """
        self.env = env

    # -----------------------------------------------------------
    # Main entrypoint
    # -----------------------------------------------------------
    def probe(self):
        """
        Process probe.
        
        Returns
        -------
        Any
            Function result.
        
        """
        script = r'''
import os, json, subprocess

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except:
        return None

# ---------------------------------------------------------------
# 1. OS INFO
# ---------------------------------------------------------------
os_release = {}
try:
    for line in open("/etc/os-release"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os_release[k] = v.strip('"')
except:
    pass

# ---------------------------------------------------------------
# 2. DESKTOP ENVIRONMENT DETECTION
# ---------------------------------------------------------------
de = os.environ.get("XDG_CURRENT_DESKTOP")

if not de:
    # Detection via tools (robust even in OSWorld)
    if run("which gsettings"):
        de = "GNOME"
    elif run("qdbus org.kde.KWin"):
        de = "KDE"
    elif run("which xfconf-query"):
        de = "XFCE"
    else:
        de = None

# ---------------------------------------------------------------
# 3. DISPLAY SERVER DETECTION
# ---------------------------------------------------------------
session_type = os.environ.get("XDG_SESSION_TYPE")

if not session_type:
    sess = run("loginctl | grep $(whoami)")
    if sess:
        sid = sess.split()[0]
        t = run(f"loginctl show-session {sid} -p Type")
        if t and "=" in t:
            session_type = t.split("=")[1]

# ---------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------
result = {
    "os": {
        "pretty_name": os_release.get("PRETTY_NAME"),
        "id": os_release.get("ID"),
        "version": os_release.get("VERSION_ID"),
    },
    "desktop_environment": de,
    "display_server": session_type,
}

print(json.dumps(result))
'''

        raw = self.env.python_exec(script)
        info = self._parse(raw)
        info["summary"] = self._summarize(info)
        return info

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------
    def _parse(self, resp):
        """
        Parse `os-prober` output into a structured dictionary.
        
        Parameters
        ----------
        resp : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        payload = resp["result"]["output"]
        return json.loads(payload)

    def _summarize(self, info):
        """
        Process summarize.
        
        Parameters
        ----------
        info : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        osname  = info.get("os", {}).get("pretty_name", "Unknown OS")
        de      = info.get("desktop_environment") or "Unknown"
        display = info.get("display_server")      or "Unknown"

        return f"""System profile:
- OS: {osname}
- Desktop environment: {de}
- Display server: {display}
- Administrator credentials:
    - username: user
    - password: password
"""
