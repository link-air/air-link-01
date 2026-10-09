' =============================================================
'  ASCII alias for the desktop on/off switch (double-click me).
'
'  The canonical switch is the Chinese-named .hta next to this
'  file -- a tiny window with two buttons: "bring everything
'  up" (= the launcher / start.vbs) and "quit gracefully" (= Settings
'  -> Quit all: voice service first, then the dashboard).
'
'  This alias just opens that .hta with mshta. Same reason as
'  start.vbs for building the name with ChrW(): this file stays
'  pure ASCII, and only the Chinese name is assembled at runtime.
' =============================================================
Option Explicit

Dim sh, fso, root, target
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName) & "\"

' U+5F00 U+5173 = the two CJK characters  ->  "<name>.hta"
target = ChrW(&H5F00) & ChrW(&H5173) & ".hta"

If Not fso.FileExists(root & target) Then
  MsgBox "Cannot find " & target & " next to this file." & vbCrLf & vbCrLf & _
         "Keep switch.vbs and " & target & " in the same folder.", _
         48, "air-link-01"
  WScript.Quit 1
End If

sh.Run "mshta.exe """ & root & target & """", 1, False
