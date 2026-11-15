# MORPHOS master CLI (T5.1)
#
# A single supervisor binary that coordinates the three toolchains of the
# platform:
#
#   morphos run        -> the Python neural core (main.py)
#   morphos ingest     -> the Rust binary event parser (morphos-trace)
#   morphos serve      -> the Node WebSocket hub + Three.js studio
#   morphos export     -> WAL/Trace -> JSON frames or CSV
#   morphos experiment -> the reproducible benchmark runners
#   morphos doctor     -> toolchain + artifact diagnosis
#   morphos init       -> scaffold a project (config + data dir)
#
# It shells out instead of linking Python: that keeps the binary small and
# means a broken toolchain produces a readable error rather than a crash.

import std/[os, osproc, strutils, strformat, times, json, sequtils, tables, algorithm]

import config, exec

const Usage = """
  MORPHOS master CLI  v{version}

  USAGE
    morphos <command> [options]

  COMMANDS
    init [dir]                  Scaffold a project (morphos.toml + data dirs)
    run [options]               Run the neural control loop
    ingest <file> [options]     Parse a binary trace with the Rust engine
    export [options]            Convert WAL/trace output to JSON or CSV
    serve [options]             Start the visualisation studio
    experiment <name> [opts]    Run a benchmark / experiment
    doctor                      Check toolchains, artifacts and the ABI
    version                     Print version information
    help                        Show this message

  RUN OPTIONS (override morphos.toml)
    --steps N                   Control steps to simulate        (default 1200)
    --neurons N                 LIF neurons in the ensemble      (default 400)
    --chunk N                   Neural sim steps per chunk       (default 200)
    --chaos none|combat|combat5 Perturbation profile            (default none)
    --source lorenz|rust|system State source                    (default lorenz)
    --trace PATH                Binary trace for --source rust
    --data-dir DIR              Output directory                 (default data)
    --resume                    Continue from the latest snapshot
    --detach                    Run in the background, log to data-dir/run.log
    --feed URL                  Stream live frames to a studio hub (T5.3)

  INGEST OPTIONS
    --json                      Machine-readable statistics
    --stream                    Emit one JSON line per event
    --hz N                      Decimation for --stream           (default 0=all)
    --mkstream                  Generate a trace instead of reading one
    --secs S                    Length for --mkstream            (default 10)

  EXPORT OPTIONS
    --format json|csv|html      Output format                    (default json)
    --out PATH                  Output file (default stdout)
    --limit N                   Max records                    (default: all)

  SERVE OPTIONS
    --port N                    Port for the studio              (default 8787)
    --source lorenz|rust|system Data source for the live hub
    --data-dir DIR              Where snapshots/WAL live
    --no-browser                Do not open a browser

  EXPERIMENT NAMES
    interop     cross-source subspace ablation (experiment_002)
    transfer    cross-domain zero-shot transfer  (experiment_003)
    anomaly     detector benchmark vs baselines  (benchmarks)

  GLOBAL
    --config PATH               Use a specific config file (default morphos.toml)
    --verbose                   Echo every child command before running it
    -h, --help                  Show this message
    -V, --version               Print the version
"""

proc printUsage(exitCode = 0) =
  echo Usage.replace("{version}", MorphosVersion)
  if exitCode != 0:
    quit(exitCode)

type ParsedArgs = object
  configPath: string
  verbose: bool
  positional: seq[string]
  flags: seq[string]
  values: seq[(string, string)]

# Switches that never take a value. Without this list, `--mkstream file.x`
# would consume the path as the flag's argument.
const BooleanFlags = ["--resume", "--detach", "--json", "--stream",
                      "--mkstream", "--no-browser", "--verbose", "--help"]

proc isBooleanSwitch(a: string): bool =
  a in BooleanFlags

proc parseArgv(argv: seq[string]): ParsedArgs =
  result.configPath = ConfigFileName
  result.positional = @[]
  result.flags = @[]
  result.values = @[]
  var i = 0
  while i < argv.len:
    let a = argv[i]
    if a == "--config":
      if i + 1 >= argv.len:
        raise newException(CliError, "--config requires a path")
      result.configPath = argv[i + 1]
      i += 2
    elif a == "--verbose":
      result.verbose = true
      i += 1
    elif a.startsWith("--"):
      # A flag takes the next token as its value unless it is a known boolean
      # switch or the next token is itself a switch; that keeps both
      # `--resume --steps 10` and `--mkstream out.x` correct.
      if not isBooleanSwitch(a) and i + 1 < argv.len and
         not argv[i + 1].startsWith("-"):
        result.values.add((a, argv[i + 1]))
        i += 2
      else:
        result.flags.add(a)
        i += 1
    else:
      result.positional.add(a)
      i += 1

proc getFlag(p: ParsedArgs, name: string, default = ""): string =
  for (k, v) in p.values:
    if k == name: return v
  default

proc hasFlag(p: ParsedArgs, name: string): bool =
  ## Named `hasFlag`, not `has`: `tables`/`json` also export `has`, and the
  ## overload silently resolved to theirs, so every `--flag` test read false.
  name in p.flags or (p.values.mapIt(it[0]).contains(name))

proc echoCmd(exe: string, args: seq[string], verbose: bool) =
  if verbose:
    var parts = @[exe]
    for a in args:
      parts.add(if a.contains(' '): "\"" & a & "\"" else: a)
    echo "  $ " & parts.join(" ")

proc runChild(exe: string, args: seq[string], verbose: bool,
              cwd = ""): int =
  ## Run a child to completion, echoing its exit status. Returns the code.
  echoCmd(exe, args, verbose)
  if exe.len == 0:
    raise newException(CliError, "required executable not found")
  let target = if cwd.len > 0: cwd else: rootDir()
  let (code, output) = runInteractive(exe, args, target)
  if output.len > 0:
    echo output
  if code != 0:
    fail(&"child exited with code {code}")
  result = code

proc applyOverrides(project: ProjectConfig, p: ParsedArgs): RunConfig =
  result = project.run
  let steps = p.getFlag("--steps")
  if steps.len > 0: result.steps = parseInt(steps)
  let neurons = p.getFlag("--neurons")
  if neurons.len > 0: result.neurons = parseInt(neurons)
  let chunk = p.getFlag("--chunk")
  if chunk.len > 0: result.chunk = parseInt(chunk)
  let chaos = p.getFlag("--chaos")
  if chaos.len > 0: result.chaos = parseChaos(chaos)
  let source = p.getFlag("--source")
  if source.len > 0: result.source = parseSource(source)
  let trace = p.getFlag("--trace")
  if trace.len > 0: result.trace = trace
  let dataDir = p.getFlag("--data-dir")
  if dataDir.len > 0: result.dataDir = dataDir
  if hasFlag(p, "--resume"): result.resume = true
  let feed = p.getFlag("--feed")
  if feed.len > 0: result.feed = feed

# ----------------------------------------------------------------------
# init
# ----------------------------------------------------------------------
proc cmdInit(p: ParsedArgs): int =
  let target = if p.positional.len > 1: p.positional[1] else: "."
  # init may point at a directory that does not exist yet.
  let dest = absolutePath(target).normalizedPath
  createDir(dest)
  let cfgPath = dest / ConfigFileName
  let projectName = dest.splitPath.tail
  banner("init -> " & dest)
  if fileExists(cfgPath):
    warn(&"{ConfigFileName} already exists, leaving it untouched")
  else:
    let project = defaultProject(projectName)
    writeFile(cfgPath, renderConfig(project, project.run))
    ok(&"wrote {cfgPath}")
  for sub in ["data", "results", "traces"]:
    let d = dest / sub
    if not dirExists(d):
      createDir(d)
      ok(&"created {d}/")
    else:
      info(&"{d}/ already present")
  # A ready-to-run example config documents the interesting switches.
  let example = dest / "morphos.rust.toml"
  if not fileExists(example):
    var rc = defaultProject(projectName)
    rc.run.source = srcRust
    rc.run.steps = 2000
    rc.run.dataDir = "data_rust"
    writeFile(example, renderConfig(rc, rc.run))
    ok(&"wrote {example} (Rust-source example)")
  echo ""
  info("next steps:")
  echo "    morphos doctor"
  echo "    morphos run --steps 400"
  echo "    morphos serve"
  0

# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
proc cmdRun(p: ParsedArgs): int =
  let cfgPath = if fileExists(p.configPath): p.configPath
                else: rootDir() / p.configPath
  let project = loadProject(cfgPath)
  let runCfg = applyOverrides(project, p)
  let py = pythonExe(project.python)
  if py.len == 0:
    raise newException(CliError,
      "Python not found. Install Python 3.10+ or set `python` in " & ConfigFileName)
  let mainPy = rootDir() / "main.py"
  if not fileExists(mainPy):
    raise newException(CliError, "main.py not found at " & mainPy)
  banner(&"run  steps={runCfg.steps} source={$runCfg.source} chaos={$runCfg.chaos}")
  info(&"python: {py}")
  info(&"data:   {runCfg.dataDir}")
  let args = @[mainPy] & argsFromRun(runCfg)
  if hasFlag(p, "--detach"):
    let logDir = rootDir() / runCfg.dataDir
    createDir(logDir)
    let logPath = logDir / "run.log"
    echoCmd(py, args, p.verbose)
    # Detach with the log redirected by the shell. Argument quoting is done
    # here so paths containing spaces survive the shell round-trip.
    let quoted = args.mapIt(
      if it.contains(' ') or it.contains(DirSep): "\"" & it & "\"" else: it)
    let cmd = "\"" & py & "\" " & quoted.join(" ") & " > \"" & logPath & "\" 2>&1"
    let launched = startProcess(cmd, workingDir = rootDir(),
                                options = {poUsePath, poEvalCommand,
                                           poParentStreams})
    launched.close()
    ok(&"started in background -> {logPath}")
    info("tail it with: morphos export --format text --out " & logPath)
    return 0
  let code = runChild(py, args, p.verbose)
  if code == 0:
    ok(&"run complete ({runCfg.steps} steps, {runCfg.steps * 5} ms sim)")
    info(&"data: {runCfg.dataDir}/trajectory.wal, {runCfg.dataDir}/morphos.db")
  code

# ----------------------------------------------------------------------
# ingest
# ----------------------------------------------------------------------
proc cmdIngest(p: ParsedArgs): int =
  let exe = rustExe()
  if exe.len == 0:
    raise newException(CliError,
      "morphos-trace not built. Run: cd engines/rust_trace && cargo build --release")
  banner("ingest")
  info(&"engine: {exe}")
  when defined(morphosDebug):
    echo "  [dbg] positional=", p.positional, " flags=", p.flags,
         " values=", p.values
  when defined(morphosDebug):
    echo "  [dbg] hasFlag(mkstream)=", hasFlag(p, "--mkstream"),
         " positional.len=", p.positional.len
  var args: seq[string]
  if hasFlag(p, "--mkstream"):
    # Resolve against the caller's directory: the child runs from the project
    # root, so a relative path like traces/x.morphose must be anchored first.
    let rawTarget = if p.positional.len > 1: p.positional[1]
                    else: "traces/generated.morphose"
    # `expandFilename` raises when the file does not exist yet; `absolutePath`
    # only needs the parent, which `createDir` makes right after.
    let target = absolutePath(rawTarget).normalizedPath
    createDir(target.parentDir)
    let secs = p.getFlag("--secs", "10")
    let hz = p.getFlag("--hz", "1000")
    let chaos = p.getFlag("--chaos", "none")
    args = @["mkstream", target, "--secs", secs, "--hz", hz,
             "--chaos", chaos]
    when defined(morphosDebug):
      echo "  [dbg] mkstream args=", args
  else:
    if p.positional.len < 2:
      raise newException(CliError, "ingest requires a trace file path")
    let target = absolutePath(p.positional[1]).normalizedPath
    if not fileExists(target):
      raise newException(CliError, "trace not found: " & target)
    if hasFlag(p, "--stream"):
      args = @["stream", target]
      let hz = p.getFlag("--hz")
      if hz.len > 0:
        args.add @["--hz", hz]
    else:
      args = @["scan", target]
      if hasFlag(p, "--json"):
        args.add "--json"
  let code = runChild(exe, args, p.verbose)
  if code == 0 and not hasFlag(p, "--stream") and not hasFlag(p, "--json"):
    ok("trace parsed")
  code

# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
proc cmdExport(p: ParsedArgs): int =
  banner("export")
  let py = pythonExe()
  if py.len == 0:
    raise newException(CliError, "Python not found")
  let script = rootDir() / "tools" / "morphos_export.py"
  if not fileExists(script):
    raise newException(CliError,
      "export tool missing: " & script & " (expected tools/morphos_export.py)")
  var args = @[script]
  let wal = p.getFlag("--wal", (if p.positional.len > 1: p.positional[1]
                                else: "data/trajectory.wal"))
  args.add @["--wal", wal]
  let fmt = p.getFlag("--format")
  if fmt.len > 0: args.add @["--format", fmt]
  let outPath = p.getFlag("--out")
  if outPath.len > 0: args.add @["--out", outPath]
  let limit = p.getFlag("--limit")
  if limit.len > 0: args.add @["--limit", limit]
  if hasFlag(p, "--trace"):
    args.add @["--trace", p.getFlag("--trace")]
  runChild(py, args, p.verbose)

# ----------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------
proc cmdServe(p: ParsedArgs): int =
  banner("serve")
  let node = nodeExe()
  if node.len == 0:
    raise newException(CliError,
      "Node.js not found. Install Node 18+ to run the visualisation studio.")
  let serverJs = rootDir() / "visualization" / "server.js"
  if not fileExists(serverJs):
    raise newException(CliError, "server missing: " & serverJs)
  let port = p.getFlag("--port", $loadProject(p.configPath).vizPort)
  var args = @[serverJs, "--port", port]
  let src = p.getFlag("--source")
  if src.len > 0: args.add @["--source", src]
  let dataDir = p.getFlag("--data-dir")
  if dataDir.len > 0: args.add @["--data-dir", dataDir]
  if hasFlag(p, "--no-browser"): args.add "--no-browser"
  ok(&"studio: http://localhost:{port}/")
  info("press Ctrl+C to stop")
  runChild(node, args, p.verbose)

# ----------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------
const Experiments = {
  "interop": "experiments/experiment_002/run.py",
  "transfer": "experiments/experiment_003/run.py",
  "anomaly": "experiments/benchmarks/runner.py",
}.toTable

proc cmdExperiment(p: ParsedArgs): int =
  if p.positional.len < 2:
    raise newException(CliError,
      "experiment requires a name: interop | transfer | anomaly")
  let name = p.positional[1]
  if name notin Experiments:
    raise newException(CliError,
      "unknown experiment '" & name & "'. Known: " &
      Experiments.keys.toSeq.sorted.join(", "))
  let py = pythonExe()
  if py.len == 0:
    raise newException(CliError, "Python not found")
  let script = rootDir() / Experiments[name]
  if not fileExists(script):
    raise newException(CliError, "experiment script missing: " & script)
  banner("experiment: " & name)
  # Pass through only the switches the runners understand.
  var args = @[script]
  for key in ["--train-steps", "--test-steps", "--calib-steps",
              "--anomaly-steps", "--out"]:
    let v = p.getFlag(key)
    if v.len > 0:
      args.add @[key, v]
  runChild(py, args, p.verbose)

# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------
proc cmdDoctor(p: ParsedArgs): int =
  banner(&"doctor  (morphos {MorphosVersion})")
  echo "  root: " & rootDir()
  echo ""
  var problems = 0

  # toolchains
  let py = pythonExe()
  let tiPy = probeTool("python", py, @["--version"], "install Python 3.10+")
  let tiRust = probeTool("rust engine", rustExe(), @["version"],
                         "cd engines/rust_trace && cargo build --release")
  let tiCargo = probeTool("cargo", cargoExe(), @["--version"],
                          "install Rust toolchain")
  let tiNode = probeTool("node", nodeExe(), @["--version"],
                         "install Node.js 18+")
  let tiNim = probeTool("nim", nimExe(), @["--version"],
                        "install Nim 2.0+ (only needed to rebuild the CLI)")
  for ti in [tiPy, tiRust, tiCargo, tiNode, tiNim]:
    if ti.found:
      ok(&"{ti.name:12} {ti.version}")
    else:
      if ti.name in ["python", "rust engine"]:
        inc problems
      warn(&"{ti.name:12} not found ({ti.hint})")

  # python imports
  echo ""
  if py.len > 0:
    let (code, output) = runCapture(py, @ ["-c",
      "import nengo,numpy;print('nengo',nengo.__version__,'numpy',numpy.__version__)"])
    if code == 0:
      ok("deps         " & output.strip())
    else:
      inc problems
      fail("deps         nengo/numpy missing -> pip install -r requirements.txt")

  # rust library + ABI
  echo ""
  let libPath = rustLibPath()
  if fileExists(libPath):
    ok(&"rust lib     {libPath}")
    if py.len > 0:
      let (code, output) = runCapture(py, @ ["-c",
        "import sys;sys.path.insert(0,'.');from core.ingest import rust_trace as r;" &
        "s=r.status();print(('OK' if s['available'] else 'UNAVAILABLE')+' '+str(s['error'] or ''))"])
      if code == 0:
        let txt = output.strip()
        if txt.startsWith("OK"):
          ok("ffi          bridge loaded, ABI 5 verified")
        else:
          inc problems
          fail("ffi          " & txt)
      else:
        inc problems
        fail("ffi          could not import the bridge")
  else:
    inc problems
    fail(&"rust lib     missing at {libPath} -> cargo build --release")

  # visualization
  echo ""
  let vizIndex = rootDir() / "visualization" / "public" / "index.html"
  if fileExists(vizIndex):
    ok("studio       visualization/public/index.html present")
  else:
    warn("studio       not built yet (run: morphos serve)")

  # project config + data
  echo ""
  let cfgPath = rootDir() / p.configPath
  if fileExists(cfgPath):
    let project = loadProject(cfgPath)
    ok(&"config       {cfgPath} (name={project.name})")
    let ddir = rootDir() / project.run.dataDir
    if dirExists(ddir):
      let wal = ddir / "trajectory.wal"
      if fileExists(wal):
        let sz = getFileSize(wal)
        ok(&"data         {wal} ({sz div 1024} KiB)")
      else:
        info(&"data         {ddir} (no trajectory.wal yet)")
    else:
      info(&"data         {ddir} will be created on first run")
  else:
    warn(&"config       {cfgPath} not found (run: morphos init)")

  echo ""
  if problems == 0:
    ok("all required components are available")
  else:
    fail(&"{problems} required component(s) missing")
  problems

# ----------------------------------------------------------------------
# version / main
# ----------------------------------------------------------------------
proc cmdVersion(): int =
  echo "morphos " & MorphosVersion
  echo "  python core : Phase IV.b (dual-opinion anomaly detection)"
  echo "  rust engine : trace ABI 5 (mmap reader + SPSC ring)"
  echo "  studio      : WebSocket hub + Three.js"
  echo "  platform    : " & hostOS & " / " & hostCPU
  0

proc main(): int =
  let argv = commandLineParams()
  if argv.len == 0:
    printUsage(1)
  if "-h" in argv or "--help" in argv or argv[0] == "help":
    printUsage(0)
  if "-V" in argv or "--version" in argv or argv[0] == "version":
    return cmdVersion()
  var p: ParsedArgs
  try:
    p = parseArgv(argv)
    case p.positional[0]
    of "init": result = cmdInit(p)
    of "run": result = cmdRun(p)
    of "ingest": result = cmdIngest(p)
    of "export": result = cmdExport(p)
    of "serve": result = cmdServe(p)
    of "experiment": result = cmdExperiment(p)
    of "doctor": result = cmdDoctor(p)
    else:
      fail("unknown command: " & p.positional[0])
      echo ""
      printUsage(2)
  except CliError as e:
    echo ""
    fail(e.msg)
    result = 2
  except ValueError as e:
    fail("invalid argument: " & e.msg)
    result = 2
  except CatchableError as e:
    fail(e.msg)
    result = 3

when isMainModule:
  quit(main())
