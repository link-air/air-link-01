' =============================================================
'  ASCII alias for the no-console launcher (double-click me).
'
'  The canonical launcher is the Chinese-named `启动.vbs` next
'  to this file. Its own name is not typeable on an English
'  keyboard, so this thin alias exists: it resolves the name at
'  runtime with ChrW() and hands over to it.
'
'  Why ChrW instead of writing the name literally: this file is
'  PURE ASCII ON PURPOSE (same reason as 启动.vbs and run.cmd --
'  wscript/cmd read these files with the system ANSI code page,
'  so multibyte characters in the source are a real hazard).
'
'  What it does is exactly 启动.vbs: start the memory workbench
'  with pythonw (no black console box) and bring the local voice
'  service up along with it; double-clicking twice will not start
'  a second dashboard.
'
'  To stop everything: dashboard -> Settings -> [Quit all]
'  (voice service first, so the VRAM goes back before the
'   dashboard itself goes down).
' =============================================================
Option Explicit

Dim sh, fso, root, target
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Where am I? Derived from this script's own path -- the folder can be moved anywhere.
root = fso.GetParentFolderName(WScript.ScriptFullName) & "\"

' U+542F U+52A8 = "启动"  ->  "启动.vbs"
target = ChrW(&H542F) & ChrW(&H52A8) & ".vbs"

If Not fso.FileExists(root & target) Then
  MsgBox "Cannot find " & target & " next to this file." & vbCrLf & vbCrLf & _
         "Keep start.vbs and " & target & " in the same folder.", _
         48, "air-link-01"
  WScript.Quit 1
End If

sh.Run "wscript.exe """ & root & target & """", 0, False
