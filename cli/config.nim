# MORPHOS CLI - shared types, config model and small utilities.
#
# Deliberately dependency-free: no nimble packages, no external runtime. The
# binary is a supervisor for the Python core and the Rust engine, so it only
# needs to parse a config file, build argv vectors and manage child processes.

import std/[os, strutils, json, tables]

const
  MorphosVersion* = "0.5.0"
  ConfigFileName* = "morphos.toml"
  DefaultDataDir* = "data"

type
  SourceKind* = enum
    srcLorenz = "lorenz"
    srcRust = "rust"
    srcSystem = "system"

  ChaosKind* = enum
    chNone = "none"
    chCombat = "combat"
    chCombat5 = "combat5"

  RunConfig* = object
    ## Everything `morphos run` needs, with defaults matching main.py.
    steps*: int
    neurons*: int
    chunk*: int
    recurrentGain*: float
    mu*: float
    source*: SourceKind
    chaos*: ChaosKind
    dataDir*: string
    trace*: string
    resume*: bool
    feed*: string

  ProjectConfig* = object
    ## On-disk project description (morphos.toml).
    name*: string
    dataDir*: string
    python*: string
    run*: RunConfig
    vizPort*: int

  CliError* = object of CatchableError
    ## Raised for user-facing failures; main() turns it into a clean message.

proc defaultRunConfig*(): RunConfig =
  RunConfig(steps: 1200, neurons: 400, chunk: 200, recurrentGain: 0.5,
            mu: 0.05, source: srcLorenz, chaos: chNone,
            dataDir: DefaultDataDir, trace: "", resume: false, feed: "")

proc defaultProject*(name = "morphos"): ProjectConfig =
  ProjectConfig(name: name, dataDir: DefaultDataDir, python: "",
                run: defaultRunConfig(), vizPort: 8787)

# ----------------------------------------------------------------------
# Minimal TOML-ish reader
#
# A full TOML parser is not needed: the config is flat `key = value` pairs
# grouped under `[sections]`. Supporting exactly that keeps the CLI
# dependency-free and the file readable. JSON is also accepted (detected by a
# leading '{') so tooling can generate configs.
# ----------------------------------------------------------------------
proc parseBool(s: string, fallback: bool): bool =
  case s.toLowerAscii()
  of "true", "yes", "on", "1": true
  of "false", "no", "off", "0": false
  else: fallback

proc unquote(s: string): string =
  var t = s.strip()
  if t.len >= 2 and ((t[0] == '"' and t[^1] == '"') or
                     (t[0] == '\'' and t[^1] == '\'')):
    t = t[1 .. ^2]
  t

proc tomlPairs*(text: string): Table[string, Table[string, string]] =
  ## Returns section -> (key -> raw value). Keys outside a section land in "".
  result = initTable[string, Table[string, string]]()
  result[""] = initTable[string, string]()
  var section = ""
  for rawLine in text.splitLines():
    var line = rawLine
    let hash = line.find('#')
    if hash >= 0:
      line = line[0 ..< hash]
    line = line.strip()
    if line.len == 0:
      continue
    if line[0] == '[' and line[^1] == ']':
      section = unquote(line[1 .. ^2]).strip().toLowerAscii()
      if section notin result:
        result[section] = initTable[string, string]()
      continue
    let eq = line.find('=')
    if eq <= 0:
      continue
    let key = unquote(line[0 ..< eq]).strip().toLowerAscii()
    let val = line[eq + 1 .. ^1].strip()
    result[section][key] = val

proc lookup(pairs: Table[string, Table[string, string]], section,
            key: string): string =
  if section in pairs and key in pairs[section]:
    return pairs[section][key]
  if "" in pairs and key in pairs[""]:
    return pairs[""][key]
  ""

proc parseSource*(s: string): SourceKind =
  try:
    result = parseEnum[SourceKind](s.strip().toLowerAscii())
  except ValueError:
    raise newException(CliError,
      "unknown source '" & s & "' (expected lorenz|rust|system)")

proc parseChaos*(s: string): ChaosKind =
  try:
    result = parseEnum[ChaosKind](s.strip().toLowerAscii())
  except ValueError:
    raise newException(CliError,
      "unknown chaos '" & s & "' (expected none|combat|combat5)")

proc loadProject*(path: string): ProjectConfig =
  ## Read morphos.toml (or a JSON file) into a ProjectConfig. Missing keys keep
  ## their defaults so a partial config is always valid.
  result = defaultProject()
  if not fileExists(path):
    return result
  let text = readFile(path)
  let trimmed = text.strip()
  if trimmed.len > 0 and trimmed[0] == '{':
    let j = parseJson(trimmed)
    if j.hasKey("name"): result.name = j["name"].getStr()
    if j.hasKey("data_dir"): result.dataDir = j["data_dir"].getStr()
    if j.hasKey("python"): result.python = j["python"].getStr()
    if j.hasKey("viz_port"): result.vizPort = j["viz_port"].getInt()
    if j.hasKey("run"):
      let r = j["run"]
      if r.hasKey("steps"): result.run.steps = r["steps"].getInt()
      if r.hasKey("neurons"): result.run.neurons = r["neurons"].getInt()
      if r.hasKey("chunk"): result.run.chunk = r["chunk"].getInt()
      if r.hasKey("recurrent_gain"):
        result.run.recurrentGain = r["recurrent_gain"].getFloat()
      if r.hasKey("mu"): result.run.mu = r["mu"].getFloat()
      if r.hasKey("source"): result.run.source = parseSource(r["source"].getStr())
      if r.hasKey("chaos"): result.run.chaos = parseChaos(r["chaos"].getStr())
      if r.hasKey("data_dir"): result.run.dataDir = r["data_dir"].getStr()
      if r.hasKey("trace"): result.run.trace = r["trace"].getStr()
      if r.hasKey("resume"): result.run.resume = r["resume"].getBool()
      if r.hasKey("feed"): result.run.feed = r["feed"].getStr()
    return result

  let pairs = tomlPairs(text)
  let nm = lookup(pairs, "", "name")
  if nm.len > 0: result.name = unquote(nm)
  let dd = lookup(pairs, "", "data_dir")
  if dd.len > 0: result.dataDir = unquote(dd)
  let py = lookup(pairs, "", "python")
  if py.len > 0: result.python = unquote(py)
  let vp = lookup(pairs, "", "viz_port")
  if vp.len > 0: result.vizPort = parseInt(unquote(vp))

  template grab(sec, key: string, field: untyped) =
    let v = lookup(pairs, sec, key)
    if v.len > 0: field = unquote(v)

  var steps, neurons, chunk, gain, mu, ddir, trace, resume, feed: string
  grab("run", "steps", steps)
  grab("run", "neurons", neurons)
  grab("run", "chunk", chunk)
  grab("run", "recurrent_gain", gain)
  grab("run", "mu", mu)
  grab("run", "data_dir", ddir)
  grab("run", "trace", trace)
  grab("run", "resume", resume)
  grab("run", "feed", feed)
  if steps.len > 0: result.run.steps = parseInt(steps)
  if neurons.len > 0: result.run.neurons = parseInt(neurons)
  if chunk.len > 0: result.run.chunk = parseInt(chunk)
  if gain.len > 0: result.run.recurrentGain = parseFloat(gain)
  if mu.len > 0: result.run.mu = parseFloat(mu)
  if ddir.len > 0: result.run.dataDir = ddir
  if trace.len > 0: result.run.trace = trace
  if resume.len > 0: result.run.resume = parseBool(resume, false)
  if feed.len > 0: result.run.feed = feed

  let src = lookup(pairs, "run", "source")
  if src.len > 0: result.run.source = parseSource(unquote(src))
  let chaos = lookup(pairs, "run", "chaos")
  if chaos.len > 0: result.run.chaos = parseChaos(unquote(chaos))

proc renderConfig*(p: ProjectConfig, run: RunConfig): string =
  ## Serialise back to morphos.toml (used by `morphos init`).
  result = ""
  result.add "# MORPHOS project configuration\n"
  result.add "# Generated by `morphos init`. Edit freely; every key is optional.\n\n"
  result.add "name = \"" & p.name & "\"\n"
  result.add "data_dir = \"" & p.dataDir & "\"\n"
  result.add "viz_port = " & $p.vizPort & "\n"
  if p.python.len > 0:
    result.add "python = \"" & p.python & "\"\n"
  result.add "\n[run]\n"
  result.add "steps = " & $run.steps & "\n"
  result.add "neurons = " & $run.neurons & "\n"
  result.add "chunk = " & $run.chunk & "\n"
  result.add "recurrent_gain = " & $run.recurrentGain & "\n"
  result.add "mu = " & $run.mu & "\n"
  result.add "source = \"" & $run.source & "\"\n"
  result.add "chaos = \"" & $run.chaos & "\"\n"
  result.add "data_dir = \"" & run.dataDir & "\"\n"
  result.add "resume = " & (if run.resume: "true" else: "false") & "\n"
  # Live studio feed (T5.3); empty means "no feed".
  result.add "feed = " & (if run.feed.len > 0: "\"" & run.feed & "\"" else: "\"\"") & "\n"

proc argsFromRun*(r: RunConfig): seq[string] =
  ## Translate a RunConfig into main.py CLI switches.
  result = @["--steps", $r.steps, "--neurons", $r.neurons,
             "--chunk", $r.chunk,
             "--recurrent-gain", $r.recurrentGain.formatFloat(ffDecimal, 4),
             "--mu", $r.mu.formatFloat(ffDecimal, 6),
             "--data-dir", r.dataDir]
  result.add @["--source", $r.source]
  if r.chaos != chNone:
    result.add @["--chaos", $r.chaos]
  if r.source == srcRust and r.trace.len > 0:
    result.add @["--trace", r.trace]
  if r.feed.len > 0:
    # Live studio feed (T5.3): main.py POSTs frames to this hub URL.
    result.add @["--feed", r.feed]
  if r.resume:
    result.add "--resume"

proc banner*(text: string) =
  echo ""
  echo "  MORPHOS  " & text
  echo "  " & repeat('-', text.len + 10)

proc ok*(msg: string) = echo "  [ok]   " & msg
proc info*(msg: string) = echo "  [..]   " & msg
proc warn*(msg: string) = echo "  [warn] " & msg
proc fail*(msg: string) = echo "  [fail] " & msg
