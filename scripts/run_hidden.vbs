' =====================================================================
' run_hidden.vbs - start a console launcher with NO visible window.
'
' WHY THIS EXISTS
' ---------------
' The scheduled task "\Agent1\Kanban Queue Drain" fires every 5 minutes
' and its action is a .cmd file.  Task Scheduler starts a .cmd through
' cmd.exe, and under LogonType=InteractiveToken that console is created
' in the USER'S session - so a black cmd.exe window pops up on the
' desktop every 5 minutes, takes focus from whatever is being typed
' into, and disappears.  A timer that steals focus four times an hour
' is unusable, and hiding the window is not something Task Scheduler
' can do on its own: it has no "hidden" flag for a console action.
'
' WScript.Shell.Run(..., 0, ...) passes intWindowStyle = 0, which asks
' Windows to CREATE THE PROCESS WITH A HIDDEN WINDOW.  The drain keeps
' its console (so cmd.exe redirection, chcp and %ERRORLEVEL% work
' exactly as before) but nothing is ever shown.
'
' WHY NOT RUN THE PYTHON MODULE DIRECTLY / AS SYSTEM
' -------------------------------------------------
'   * pythonw.exe has no console, so the launcher's `>>` logging,
'     chcp 65001 and PYTHONIOENCODING would be lost.
'   * Running as SYSTEM (session 0) also hides the window, but then the
'     queue files, offset.txt and the log are written as SYSTEM, which
'     breaks ownership of files the interactive agent must also write.
' The wrapper keeps the SAME user and the SAME launcher - it only
' removes the window.
'
' EXIT CODE
' ---------
' bWaitOnReturn = True makes Run() block until the launcher exits and
' return its exit code, which we pass on with WScript.Quit.  Task
' Scheduler's "Last Run Result" therefore still shows the drain's real
' status (0 = drained/empty, 9009 = missing python, ...) instead of a
' useless 0 from an immediately-detached wscript.exe.
'
' USAGE (task action)
'   wscript.exe //B //Nologo "C:\Dev\Agent1\scripts\run_hidden.vbs"
'   wscript.exe //B //Nologo "C:\Dev\Agent1\scripts\run_hidden.vbs" "<other.cmd>"
'
' //B = batch mode, suppress script errors and prompts (a modal dialog
' from a hidden script would block the task forever with no way to see
' it).  The optional argument lets the same wrapper hide any launcher.
' =====================================================================

Option Explicit

' --- Absolute paths only (a task gets a different PATH and cwd) --------
Const DEFAULT_LAUNCHER = "C:\Dev\Agent1\scripts\run_kanban_queue_drain.cmd"
Const LOG_DIR          = "C:\Dev\Agent1\reports\harnessfix"
Const LOG_FILE         = "C:\Dev\Agent1\reports\harnessfix\kanban_queue_drain.log"

Const HIDDEN_WINDOW     = 0
Const WAIT_FOR_LAUNCHER = True
Const MISSING_LAUNCHER  = 9009

Dim shell, fso, launcher, command, rc

Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")

' --bWaitOnReturn needs the launcher path as a STRING; an empty or
' missing argument must fall back to the default, never to "".
launcher = DEFAULT_LAUNCHER
If WScript.Arguments.Count > 0 Then
    If Trim(WScript.Arguments(0)) <> "" Then launcher = WScript.Arguments(0)
End If

' A missing launcher would otherwise surface as an invisible
' 0x80070002/0x2 with no explanation anywhere.
If Not fso.FileExists(launcher) Then
    Call AppendLog("[hidden] FATAL: launcher not found at """ & launcher & """")
    WScript.Quit MISSING_LAUNCHER
End If

' Quote the path so a space in it cannot split the command line.
command = """" & launcher & """"

rc = shell.Run(command, HIDDEN_WINDOW, WAIT_FOR_LAUNCHER)

WScript.Quit rc


' Write one line to the drain log.  Only used for failures that happen
' BEFORE the launcher runs, so the launcher's own start/exit lines stay
' the normal record of a tick.
Sub AppendLog(text)
    Dim stream, stamp
    On Error Resume Next
    If Not fso.FolderExists(LOG_DIR) Then fso.CreateFolder LOG_DIR
    Set stream = fso.OpenTextFile(LOG_FILE, 8, True)
    stamp = Year(Now) & "-" & Right("0" & Month(Now), 2) & "-" & _
            Right("0" & Day(Now), 2) & " " & Right("0" & Hour(Now), 2) & _
            ":" & Right("0" & Minute(Now), 2) & ":" & Right("0" & Second(Now), 2)
    stream.WriteLine "[" & stamp & "] " & text
    stream.Close
End Sub
