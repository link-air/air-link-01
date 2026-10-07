' =============================================================
'  Start air-link-01 with NO console window (double-click me).
'
'  - starts the dashboard with pythonw (the same interpreter as
'    python, just without the black console box)
'  - asks it to bring the local voice service up too (--with-voice)
'  - if it is ALREADY running, just opens the page and makes sure
'    the voice service is up -- double-clicking twice must not try
'    to start a second dashboard (it would only die on the busy
'    port, invisibly, and look like "nothing happened")
'
'  To stop everything:  dashboard -> settings -> [全退出]
'  (that stops the voice service first, so the VRAM is released
'   before the dashboard itself goes down).
'
'  PURE ASCII ON PURPOSE: wscript reads .vbs with the system ANSI
'  code page, so multibyte characters in here are a real hazard
'  (the same reason run.cmd is ASCII-only).
' =============================================================
Option Explicit

Dim sh, fso, root, py, probe
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Where am I? Derived from this script's own path, so the folder can be
' moved or copied anywhere without editing this file.
root = fso.GetParentFolderName(WScript.ScriptFullName)

' The interpreter: the same one run.cmd falls back to (candidate 2).
' The python.exe on PATH is the WindowsApps stub -- it cannot run this.
py = sh.ExpandEnvironmentStrings("%USERPROFILE%") _
     & "\.workbuddy\binaries\python\versions\3.14.3\pythonw.exe"

' Already up? Then just show the page and nudge the voice service.
' GET first (read-only, safe to aim at whatever answers), then the POST.
' Short timeouts so a half-dead port (a dashboard still doing its exit
' flush) cannot freeze this launcher: better to fail the probe and go
' down the normal start path than to hang here silently.
On Error Resume Next
Set probe = CreateObject("MSXML2.ServerXMLHTTP")
probe.setTimeouts 1500, 1500, 2500, 1500
probe.open "GET", "http://127.0.0.1:8765/api/state", False
probe.send
If Err.Number = 0 Then
  probe.open "POST", "http://127.0.0.1:8765/api/voice/start", False
  probe.setRequestHeader "Content-Type", "application/json"
  probe.send "{}"
  sh.Run "http://127.0.0.1:8765/", 1, False
  WScript.Quit 0
End If
Err.Clear
On Error GoTo 0

If Not fso.FileExists(py) Then
  MsgBox "Not found:" & vbCrLf & py & vbCrLf & vbCrLf _
       & "Edit this file and point the py line at your pythonw.exe.", 16, "air-link-01"
  WScript.Quit 1
End If
If Not fso.FileExists(fso.BuildPath(root, "core\dashboard.py")) Then
  MsgBox "Not found:" & vbCrLf & fso.BuildPath(root, "core\dashboard.py") & vbCrLf & vbCrLf _
       & "Keep this file in the project root.", 16, "air-link-01"
  WScript.Quit 1
End If

sh.CurrentDirectory = root
' 2nd arg 0 = hidden window, 3rd arg False = do not wait for it
sh.Run """" & py & """ -m core.dashboard --with-voice", 0, False
