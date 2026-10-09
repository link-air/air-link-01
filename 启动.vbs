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
'  To stop everything:  dashboard -> settings -> [Quit all]
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

' The interpreter, in order of preference:
'   1) pythonw.exe / python.exe resolved on PATH, skipping the WindowsApps
'      stub (that entry is not Python -- it just opens the Store);
'   2) this dev machine's known install path (fallback for a machine whose
'      PATH is not set up).
' Resolved by NAME via `where`, never by *running* a candidate: see the note
' on FindInterpreter below for why running one is not a usable probe here.
py = FindInterpreter()

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

If py = "" Then
  MsgBox "No usable Python found." & vbCrLf & vbCrLf _
       & "Install Python 3.11+ and make sure pythonw.exe is on PATH," & vbCrLf _
       & "or edit this file and put your path in FindInterpreter().", 16, "air-link-01"
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


' Pick the interpreter: the first usable hit on PATH, else the known local
' install path, else "" (the caller shows a clear message).
'
' Why resolve by name with `where` instead of *running* each candidate and
' looking at the exit code (the way run.cmd's :try does): a GUI-subsystem
' pythonw does not report a meaningful exit code through WshShell.Run (it
' comes back 1 for a perfectly good interpreter, measured 2026-10-09), and
' invoking the WindowsApps stub would open the Store and hang this script
' behind that window. `where` + FileExists has neither problem.
Function FindInterpreter()
  Dim cands, i, ex, p, fallback
  FindInterpreter = ""
  cands = Array("pythonw.exe", "python.exe")
  For i = 0 To UBound(cands)
    Set ex = sh.Exec(sh.ExpandEnvironmentStrings("%ComSpec%") _
                     & " /c where " & cands(i) & " 2>nul")
    Do While Not ex.StdOut.AtEndOfStream
      p = Trim(ex.StdOut.ReadLine())
      If Len(p) > 0 And InStr(LCase(p), "windowsapps") = 0 Then
        If fso.FileExists(p) Then
          FindInterpreter = p
          Exit Function
        End If
      End If
    Loop
  Next
  fallback = sh.ExpandEnvironmentStrings("%USERPROFILE%") _
    & "\.workbuddy\binaries\python\versions\3.14.3\pythonw.exe"
  If fso.FileExists(fallback) Then FindInterpreter = fallback
End Function
