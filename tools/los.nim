## MORPHOS one-click launcher (T5.1 / Phase V final).
##
## A single tiny binary that opens an interactive menu and dispatches to the
## master CLI (bin/morphos.exe) or directly to the parts, so a new user can go
## from "nothing installed" to "3D studio streaming a live run" without
## remembering any flags.
##
## Deliberately dumb: no config parsing beyond delegating, no hidden state.
## Everything it can do, "morphos <cmd>" can do too - this is a convenience
## front door, not a second implementation.
##
## Build:  nim c -d:release -o:bin/los.exe tools/los.nim

import std/[os, osproc, strutils, terminal]

const
  Version = "1.0.0"
  Banner = """
  ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗ ██████╗ ███████╗
  ████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║██╔═══██╗██╔════╝
  ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║██║   ██║███████╗
  ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║██║   ██║╚════██║
  ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║╚██████╔╝███████║
  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

# ----------------------------------------------------------------------
# locating things
# ----------------------------------------------------------------------
proc rootDir(): string =
  ## Walk up from the executable until cli/ and core/ exist.
  var d = getAppDir()
  for _ in 0 ..< 5:
    if dirExists(d / "cli") and dirExists(d / "core"):
      return d
    let parent = d.parentDir()
    if parent == d: break
    d = parent
  return getAppDir()

proc findExe(name: string): string =
  ## Search next to us, then bin/, then PATH.
  let root = rootDir()
  for cand in [getAppDir() / name,
               root / "bin" / name,
               root / name,
               root / "target" / "release" / name]:
    if fileExists(cand):
      return cand
  let (outp, code) = execCmdEx(
    (if defined(windows): "where " else: "which ") & name)
  if code == 0:
    for line in outp.splitLines():
      let t = line.strip()
      if t.len > 0 and fileExists(t):
        return t
  return ""

proc pythonExe(): string =
  ## Find a Python with the project's core runtime dependencies.
  let probe = "import nengo, numpy"
  var cands = @["python", "python3", "py -3.12", "py -3.11", "py -3.13",
                "py -3.10"]
  # Prefer a venv inside the project if present.
  let root = rootDir()
  for v in [".venv/Scripts/python.exe", "venv/Scripts/python.exe",
            ".venv/bin/python", "venv/bin/python"]:
    if fileExists(root / v):
      cands.insert(root / v, 0)
  for c in cands:
    let (_, code) = execCmdEx(c & " -c \"" & probe & "\" 2>&1")
    if code == 0:
      return c
  return ""

proc morphosExe(): string =
  result = findExe(if defined(windows): "morphos.exe" else: "morphos")
  if result.len == 0:
    result = findExe("morphos")

# ----------------------------------------------------------------------
# small ui helpers
# ----------------------------------------------------------------------
proc hr(ch = '-', n = 62) =
  stdout.styledWriteLine(fgCyan, ch.repeat(n))

proc title(s: string) =
  echo ""
  hr('=')
  stdout.styledWriteLine(styleBright, fgCyan, "  MORPHOS  ", resetStyle,
                         fgWhite, s)
  hr('=')

proc ok(s: string) = stdout.styledWriteLine(fgGreen, "  [ok]   ", resetStyle, s)
proc info(s: string) = stdout.styledWriteLine(fgCyan, "  [..]   ", resetStyle, s)
proc warn(s: string) = stdout.styledWriteLine(fgYellow, "  [warn] ", resetStyle, s)
proc fail(s: string) = stdout.styledWriteLine(fgRed, "  [fail] ", resetStyle, s)

proc ask(prompt: string, default: string): string =
  stdout.styledWrite(fgWhite, "  " & prompt & " [" & default & "]: ")
  let line = stdin.readLine()
  let t = line.strip()
  if t.len == 0: default else: t

proc pause() =
  stdout.styledWrite(fgCyan, "\n  press Enter to return to the menu ")
  discard stdin.readLine()

# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------
proc shellRun(cmd: string): (string, int) =
  ## execCmdEx does not go through a shell on Windows, so "cd ... && x" fails.
  ## Route explicitly through the platform shell.
  when defined(windows):
    result = execCmdEx("cmd /c \"" & cmd & "\"")
  else:
    result = execCmdEx("/bin/sh -c " & quoteShell(cmd))

proc runMorphos(args: openArray[string]): int =
  let exe = morphosExe()
  if exe.len == 0:
    fail("bin/morphos.exe not found -- build it first (see menu 9)")
    return 127
  var cmd = quoteShell(exe)
  for a in args:
    cmd.add " " & quoteShell(a)
  let (outp, code) = shellRun(cmd)
  if outp.len > 0: stdout.write(outp)
  return code

proc runPython(args: openArray[string]): int =
  let py = pythonExe()
  if py.len == 0:
    fail("no Python with nengo/numpy found")
    return 127
  # cd into the project root so relative imports resolve.
  var parts = @[py]
  for a in args: parts.add(quoteShell(a))
  var full = parts.join(" ")
  let (outp, code) = shellRun("cd " & quoteShell(rootDir()) & " && " & full)
  if outp.len > 0: stdout.write(outp)
  return code

# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------
proc modeInitialise() =
  title("initialise")
  info("writing project config and directories")
  discard runMorphos(["init"])
  discard runMorphos(["doctor"])
  ok("project ready -- try 'run a simulation' next")

proc modeDoctor() =
  title("doctor")
  discard runMorphos(["doctor"])

proc modeRun() =
  title("run a simulation")
  let steps = ask("steps", "1200")
  let chaos = ask("chaos profile (none/combat)", "none")
  var args = @["run", "--steps", steps]
  if chaos != "none": args.add(["--chaos", chaos])
  discard runMorphos(args)
  ok("results in data/trajectory.wal  ·  export with menu 4")

proc modeStudio() =
  title("3D studio")
  let port = ask("port", "8787")
  let exe = morphosExe()
  if exe.len == 0:
    fail("bin/morphos.exe not found")
    return
  info("starting the studio on port " & port)
  info("open http://localhost:" & port & "/ in a browser")
  info("press Ctrl+C to stop")
  hr()
  discard runMorphos(["serve", "--port", port])

proc modeLive() =
  title("run + stream into the studio")
  let steps = ask("steps", "800")
  let port = ask("studio port", "8787")
  let exe = morphosExe()
  if exe.len == 0:
    fail("bin/morphos.exe not found")
    return
  let url = "http://localhost:" & port
  info("1/2 starting the studio in a new window ...")
  if defined(windows):
    discard execShellCmd("start \"MORPHOS studio\" /min cmd /c \"\"" &
                         quoteShell(exe) & "\" serve --port " & port & "\"")
  else:
    discard startProcess("/bin/sh", args = ["-c",
      quoteShell(exe) & " serve --port " & port & " >/dev/null 2>&1 &"],
      options = {poUsePath})
  sleep(1800)
  info("2/2 streaming a live run at " & url & " ...")
  discard runMorphos(["run", "--steps", steps, "--feed", url])
  ok("run finished -- the studio keeps the frames, explore and fork branches")

proc modeExport() =
  title("export results")
  let fmt = ask("format (json/ndjson/csv/html)", "html")
  var outName = "results/run." & fmt
  if fmt == "json": outName = "results/run.frames.json"
  discard runMorphos(["export", "--format", fmt, "--out", outName])
  ok("open " & outName)

proc modeIngest() =
  title("rust trace ingest")
  # `morphos ingest` takes the trace path positionally; --secs/--hz control
  # the generated stream.
  const Trace = "traces/demo.morphose"
  info("1) generate a synthetic trace -> " & Trace)
  discard runMorphos(["ingest", "--mkstream", Trace, "--secs", "2",
                      "--hz", "5000"])
  info("2) scan it (throughput + bounds)")
  discard runMorphos(["ingest", Trace])

proc modeBench() =
  title("benchmarks")
  info("rust crate unit tests")
  let rt = rootDir() / "engines" / "rust_trace"
  if dirExists(rt):
    let (o1, c1) = shellRun("cd " & quoteShell(rt) & " && cargo test --release 2>&1")
    stdout.write(o1)
    if c1 == 0: ok("rust tests passed") else: fail("rust tests failed")
    info("cpu benchmark")
    let (o2, _) = shellRun("cd " & quoteShell(rt) &
                           " && cargo run --release --bin morphos-cpubench 2>&1")
    stdout.write(o2)
  else:
    warn("engines/rust_trace missing")
  info("python test suites")
  for t in ["tests/test_phase5_rust.py", "tests/test_phase5_sdk.py"]:
    if fileExists(rootDir() / t):
      discard runPython([t])

proc modeExamples() =
  title("reproducible examples")
  discard runPython(["examples/run_examples.py"])

proc modeTests() =
  title("run all checks")
  let py = pythonExe()
  var failed = 0
  info("adapter SDK conformance")
  if runPython(["-m", "adapters.sdk", "conformance"]) != 0: inc failed
  info("rust bridge tests")
  if runPython(["tests/test_phase5_rust.py"]) != 0: inc failed
  info("SDK tests")
  if runPython(["tests/test_phase5_sdk.py"]) != 0: inc failed
  info("examples")
  if runPython(["examples/run_examples.py"]) != 0: inc failed
  hr()
  if failed == 0: ok("all checks passed") else: fail($failed & " suite(s) failed")

proc modeBuild() =
  title("build everything")
  let root = rootDir()
  info("1/3 building the Nim master CLI")
  let (o1, c1) = shellRun("cd " & quoteShell(root) &
    " && nim c -f -d:release --hints:off -o:bin/morphos.exe cli/morphos.nim 2>&1")
  if c1 == 0: ok("bin/morphos.exe") else:
    fail("nim build failed"); stdout.write(o1)

  info("2/3 building the Rust trace engine")
  let rt = root / "engines" / "rust_trace"
  if dirExists(rt):
    let (o2, c2) = shellRun("cd " & quoteShell(rt) &
                            " && cargo build --release 2>&1")
    if c2 == 0: ok("engines/rust_trace/target/release") else:
      fail("cargo build failed"); stdout.write(o2)
  else:
    warn("engines/rust_trace missing")

  info("3/3 checking the studio assets")
  for f in ["visualization/server.js", "visualization/public/index.html",
            "visualization/public/studio.js",
            "visualization/public/vendor/three.module.min.js"]:
    if fileExists(root / f): ok(f) else: fail("missing " & f)

proc showStatus() =
  title("status")
  let root = rootDir()
  let py = pythonExe()
  if py.len > 0: ok("python  : " & py) else: fail("python  : no nengo/numpy")
  let mx = morphosExe()
  if mx.len > 0: ok("morphos : " & mx) else: fail("morphos : not built")
  # DLLs are not on PATH; look where the Rust build actually puts them.
  var dll = ""
  for cand in [root / "engines" / "rust_trace" / "target" / "release" /
                 (if defined(windows): "rust_trace.dll" else: "librust_trace.so"),
               root / "core" / "ingest" / "native" /
                 (if defined(windows): "rust_trace.dll" else: "librust_trace.so")]:
    if fileExists(cand):
      dll = cand
      break
  if dll.len > 0: ok("rust    : " & dll) else: warn("rust    : not built")
  for f in ["morphos.toml", "visualization/server.js",
            "visualization/public/vendor/three.module.min.js"]:
    if fileExists(root / f): ok("asset   : " & f) else: warn("missing : " & f)

# ----------------------------------------------------------------------
# menu
# ----------------------------------------------------------------------
const Items = [
  ("1", "initialise project"),
  ("2", "status / doctor"),
  ("3", "run a simulation"),
  ("4", "export results"),
  ("5", "open the 3D studio"),
  ("6", "run + stream live into the studio"),
  ("7", "rust trace ingest benchmark"),
  ("8", "benchmarks"),
  ("9", "build everything"),
  ("e", "reproducible examples"),
  ("t", "run all checks"),
  ("q", "quit"),
]

proc printMenu() =
  echo ""
  hr('=')
  stdout.styledWriteLine(styleBright, fgCyan, "  MORPHOS ", resetStyle,
                         fgWhite, "v" & Version &
                         "  ·  neuro-inspired state observer")
  hr('=')
  for (k, label) in Items:
    stdout.styledWriteLine(fgYellow, "   " & k.alignLeft(2), resetStyle, label)
  hr()

proc interactive() =
  stdout.styledWriteLine(styleBright, fgCyan, Banner)
  stdout.styledWriteLine(fgWhite, "  one-click launcher  ·  v" & Version)
  stdout.styledWriteLine(fgCyan, "  project: ", resetStyle, rootDir())
  while true:
    printMenu()
    stdout.styledWrite(fgYellow, "  choose> ", resetStyle)
    var choice: string
    if not stdin.readLine(choice): break
    choice = choice.strip().toLowerAscii()
    case choice
    of "1": modeInitialise(); pause()
    of "2": showStatus(); modeDoctor(); pause()
    of "3": modeRun(); pause()
    of "4": modeExport(); pause()
    of "5": modeStudio()
    of "6": modeLive(); pause()
    of "7": modeIngest(); pause()
    of "8": modeBench(); pause()
    of "9": modeBuild(); pause()
    of "e": modeExamples(); pause()
    of "t": modeTests(); pause()
    of "q", "quit", "exit": break
    else:
      warn("unknown choice: " & choice)
      sleep(600)

proc usage() =
  echo """
MORPHOS launcher v""" & Version & """

  los                    interactive menu
  los <mode> [args...]   run a mode directly
  los --help             this text

modes
  init          initialise project config and directories
  status        report what is installed and where
  doctor        full environment probe
  run           run a simulation
  studio        open the 3D studio
  live          run + stream live frames into the studio
  export        export the last run
  ingest        rust trace ingest benchmark
  bench         run benchmarks
  build         rebuild the Nim CLI and Rust engine
  examples      run the reproducible examples
  test          run all check suites
  version       print the version

Any other arguments are forwarded to `morphos`, e.g. `los run --steps 2000`.
"""

proc main(): int =
  var args = commandLineParams()
  if args.len == 0:
    interactive()
    return 0
  let cmd = args[0].toLowerAscii()
  let rest = if args.len > 1: args[1 .. ^1] else: @[]
  case cmd
  of "-h", "--help", "help": usage(); 0
  of "-v", "--version", "version":
    echo "MORPHOS launcher " & Version; 0
  of "menu", "interactive": interactive(); 0
  of "init": modeInitialise(); 0
  of "status": showStatus(); 0
  of "doctor": modeDoctor(); 0
  of "run": modeRun(); 0
  of "studio", "serve": modeStudio(); 0
  of "live": modeLive(); 0
  of "export": modeExport(); 0
  of "ingest": modeIngest(); 0
  of "bench": modeBench(); 0
  of "build": modeBuild(); 0
  of "examples": modeExamples(); 0
  of "test", "check": modeTests(); 0
  else:
    # forward anything unknown to the master CLI
    runMorphos(args)

when isMainModule:
  quit(main())
