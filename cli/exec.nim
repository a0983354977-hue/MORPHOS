# Process execution and toolchain discovery.
#
# The CLI's job is to supervise three toolchains (Python core, Rust engine,
# Node visualisation). Everything here is about finding them reliably and
# invoking them without a shell, so quoting bugs on paths with spaces or CJK
# characters cannot creep in.

import std/[os, osproc, strutils, streams, strformat, times, sequtils]

import config

type
  ToolInfo* = object
    name*: string
    found*: bool
    version*: string
    path*: string
    hint*: string

proc rootDir*(): string =
  ## The project root, derived from the executable's location when built into
  ## <root>/bin/, otherwise the current working directory.
  let exeDir = getAppDir()
  if fileExists(exeDir / ".." / "main.py"):
    return expandFilename(exeDir / "..")
  if fileExists(exeDir / "main.py"):
    return exeDir
  if fileExists(getCurrentDir() / "main.py"):
    return getCurrentDir()
  exeDir

proc dataPath*(relative: string): string =
  rootDir() / relative

proc searchPath(name: string): string =
  ## Scan PATH for an executable, honouring PATHEXT-ish suffixes on Windows.
  let exts = if defined(windows): @["", ".exe", ".cmd", ".bat"] else: @[""]
  for dir in getEnv("PATH").split(PathSep):
    if dir.len == 0: continue
    for ext in exts:
      let candidate = dir / (name & ext)
      if fileExists(candidate):
        return expandFilename(candidate)
  ""

proc locateExe(name: string, extra: openArray[string] = []): string =
  ## Explicit candidate paths first (newest build output wins), then PATH.
  for p in extra:
    if fileExists(p):
      return expandFilename(p)
  let located = searchPath(name)
  if located.len > 0:
    return located
  ""

proc runCapture*(exe: string, args: seq[string], timeoutMs = 8000):
    tuple[code: int, output: string] =
  ## Run a command and capture stdout+stderr. Never raises for a nonzero exit.
  try:
    let p = startProcess(exe, args = args, options = {poUsePath, poStdErrToStdOut})
    var outStr = ""
    let t0 = epochTime()
    while true:
      if p.outputStream.atEnd():
        if not p.running(): break
        if (epochTime() - t0) * 1000 > timeoutMs.float:
          p.terminate()
          break
        sleep(10)
        continue
      outStr.add p.outputStream.readLine()
      outStr.add "\n"
    result.code = p.waitForExit()
    result.output = outStr
    p.close()
  except OSError as e:
    result.code = -1
    result.output = e.msg

proc shellQuote(s: string): string =
  ## Quote one argv element for the platform shell. Only the characters that
  ## actually need it are quoted, so ordinary paths stay readable in logs.
  if s.len == 0:
    return "\"\""
  const special = {' ', '\t', '"', '\'', '&', '|', '<', '>', '^', '(', ')'}
  var needs = false
  for c in s:
    if c in special:
      needs = true
      break
  if not needs:
    return s
  if defined(windows):
    # cmd.exe: wrap in double quotes and escape any embedded ones.
    result = "\""
    for c in s:
      if c == '"':
        result.add '\\'
        result.add '"'
      else:
        result.add c
    result.add '"'
  else:
    result = "'" & s.replace("'", "'\\''") & "'"

proc runInteractive*(exe: string, args: seq[string], cwd = ""):
    tuple[code: int, output: string] =
  ## Run a long-lived command, streaming its streams straight to the console.
  # osproc forbids passing args together with poEvalCommand, so a working
  # directory forces the command-string form; otherwise argv is used directly
  # (which avoids any shell quoting subtleties).
  try:
    var p: Process
    if cwd.len > 0:
      let parts = @[shellQuote(exe)] & args.mapIt(shellQuote(it))
      p = startProcess(parts.join(" "), workingDir = cwd,
                       options = {poParentStreams, poUsePath, poEvalCommand})
    else:
      p = startProcess(exe, args = args,
                       options = {poParentStreams, poUsePath})
    result.code = p.waitForExit()
    result.output = ""
    p.close()
  except OSError as e:
    result.code = -1
    result.output = e.msg

proc canImportCore*(py: string): bool =
  ## True when this interpreter can import the neural core's dependencies.
  ## A machine often has several Pythons and only one of them has nengo, so
  ## picking `python` blindly is how "deps missing" false alarms happen.
  if py.len == 0: return false
  let (code, _) = runCapture(py, @["-c", "import nengo, numpy"], 15000)
  code == 0

proc pythonExe*(override = ""): string =
  ## Resolve the interpreter that actually has the core dependencies.
  ## Order: explicit override, project venv, PATH candidates that pass the
  ## import probe, and finally the first interpreter found at all (so the
  ## caller can still produce a useful "install these" error).
  if override.len > 0 and locateExe(override).len > 0:
    return locateExe(override)
  let root = rootDir()
  for rel in [".venv/Scripts/python.exe", ".venv/bin/python",
              "venv/Scripts/python.exe", "venv/bin/python"]:
    let p = root / rel
    if fileExists(p): return expandFilename(p)

  var candidates: seq[string] = @[]
  for name in ["python3", "python"]:
    let found = searchPath(name)
    if found.len > 0: candidates.add found
  # The Windows launcher can reach every registered version; prefer the ones
  # that have the deps rather than merely the newest.
  let pyLauncher = searchPath("py")
  if pyLauncher.len > 0:
    for ver in ["-3.12", "-3.11", "-3.13", "-3.10"]:
      let (code, outStr) = runCapture(pyLauncher, @[ver, "-c",
        "import sys;print(sys.executable)"], 8000)
      if code == 0:
        let exePath = outStr.strip()
        if exePath.len > 0 and fileExists(exePath):
          candidates.add exePath

  var firstSeen = ""
  for c in candidates:
    if firstSeen.len == 0: firstSeen = c
    if canImportCore(c):
      return c
  if firstSeen.len > 0: return firstSeen
  ""

proc rustExe*(): string =
  ## The `morphos-trace` CLI, or "" when the engine is not built.
  locateExe("morphos-trace", [
    rootDir() / "engines" / "rust_trace" / "target" / "release" / "morphos-trace.exe",
    rootDir() / "engines" / "rust_trace" / "target" / "release" / "morphos-trace"])

proc rustLibPath*(): string =
  rootDir() / "engines" / "rust_trace" / "target" / "release" / "rust_trace.dll"

proc nodeExe*(): string = locateExe("node")
proc cargoExe*(): string = locateExe("cargo")
proc nimExe*(): string = locateExe("nim")

proc probeTool*(name: string, exe: string, args: seq[string],
                hint: string): ToolInfo =
  result = ToolInfo(name: name, found: exe.len > 0, path: exe, hint: hint)
  if exe.len == 0:
    return
  let (code, output) = runCapture(exe, args)
  if code == 0:
    let lines = output.splitLines()
    let firstLine = if lines.len > 0: lines[0].strip() else: ""
    result.version = if firstLine.len > 0: firstLine else: "ok"
  else:
    result.found = false
    result.version = "error (exit " & $code & ")"
