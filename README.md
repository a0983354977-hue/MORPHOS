# PROJECT MORPHOS

**A local-first, Nengo-based neural dynamics and telemetry platform.**

MORPHOS combines a Python neural-processing core, a validated local TCP/JSONL gateway, append-only WAL and SQLite persistence, and a React telemetry dashboard. The core and browser bridge run as separate processes. The browser receives persisted/core telemetry through the local bridge; the Studio dashboard does not generate mock telemetry.

> **Project status:** The Windows one-file build and the local IPC/HTTP/WebSocket flow have been exercised. Linux and macOS launch scripts and a portable PyInstaller spec are included, but native binaries must be built and tested on their target operating systems.
>
> **Scope note:** `studio/` is the current local telemetry dashboard. `visualization/` contains an older, separate visualization/demo and is not served by the MORPHOS Studio launcher.
>
> **License:** This repository currently has no `LICENSE` file. No license to reuse, modify, or redistribute the project is granted by this README.

## Contents

- [English](#english)
- [中文說明](#中文說明)

# English

## What it does

- Runs a 3-dimensional, 5 ms-step signal pipeline with a real Nengo simulator (`NengoDynamics`), feature encoding, an online NLMS predictor, anomaly scoring, and a circuit breaker.
- Accepts validated data and control requests over a loopback TCP socket using newline-delimited JSON (JSONL).
- Persists trajectory records to an append-only binary WAL and run/events metadata to SQLite in WAL mode.
- Exposes a local FastAPI HTTP/WebSocket bridge. The default bridge pushes telemetry at 20 Hz and provides paged reads of the actual WAL and SQLite data.
- Serves a React/TypeScript dashboard with signal charts, a history scrubber, a virtualized event log, host metrics, and core controls.
- Can run a deterministic Lorenz-63 source into the core. Real host CPU, memory, disk, and network metrics are available as separate bridge telemetry.
- Includes thread-safety, disconnect-isolation, restart/WAL, bridge, and packaged-launcher tests.

### Data integrity and interpretation

- The Studio charts and event list are populated from bridge telemetry, the binary WAL, and SQLite. An empty store is shown as empty; the dashboard does not fill gaps with synthetic records.
- `combat5` deliberately applies the configured chaos injector to incoming core samples. Use `live` for unmodified input.
- The Lorenz source sends real samples from the local Lorenz-63 numerical integrator to the core.
- Host resource measurements are real `psutil` readings, but are monitored separately; the current 3-D MORPHOS runtime does **not** fuse the 4-D host vector into the Nengo input.
- “Basis cross-talk” is the overlap computed from the configured orthogonal projection matrices. It is a geometric basis check, not an end-to-end measurement of two streams inside one neural network.
- Counterfactual branches replay retained in-memory core history. Only the most recent 1,024 processed states are retained for branch creation; a request outside that window is rejected.
- The older `visualization/` application is not the source of data for the current Studio dashboard.

## Architecture

```text
Lorenz-63 adapter ───────────────┐
External JSONL clients ──────────┼──> IPC daemon (Nengo + prediction + anomaly)
                                │              │
                                │              ├──> trajectory.wal
                                │              └──> morphos.db (SQLite/WAL)
                                │
Browser <── HTTP/WebSocket <── Web Bridge ─────┴── read-only history/events
                                  │
                                  └── local psutil host metrics
```

The supervisor starts two child processes:

1. **IPC daemon:** owns `MORPHOSRuntime`, Nengo, the input aligner, predictor, anomaly logic, and writers.
2. **Web Bridge:** polls the daemon over IPC, reads WAL/SQLite history, and serves the built Studio files.

Closing a browser tab does not stop either service. Stopping the supervisor requests shutdown of both child services.

## Requirements

- Python 3.10 or later for source mode.
- Node.js 18 or later only when rebuilding the Studio frontend.
- Windows, Linux, or macOS for source mode. PyInstaller output is platform-specific; build on the operating system where the executable will run.

## Quick start

### Windows — source mode

```powershell
cd MORPHOS
python -m pip install -r requirements.txt
npm --prefix studio install
npm --prefix studio run build
python studio_launcher.py
```

The launcher opens `http://127.0.0.1:8080`. To use the convenience script, run `start_morphos.bat` after building the frontend.

### Windows — standalone executable

Build the frontend and executable as described under [Build and launch](#build-and-launch). Then double-click `start_morphos.bat` or run:

```powershell
.\dist\MORPHOS_Studio.exe
```

The packaged executable includes Python, Nengo, the required Nengo data tables, the Web Bridge, and the compiled Studio assets. Node.js and a separately installed Python are not needed to run the packaged build.

### Linux and macOS — source mode

```sh
python3 -m pip install -r requirements.txt
npm --prefix studio install
npm --prefix studio run build
chmod +x start_morphos.sh
./start_morphos.sh
```

The browser opens at `http://127.0.0.1:8080`. For an OS-native standalone binary, build `morphos.spec` on that OS.

## Build and launch

### Build the frontend

```sh
npm --prefix studio install
npm --prefix studio run build
```

The output is `studio/dist/`. The Python Bridge serves this static build, so Node.js is not needed at runtime.

### Build a standalone executable

Install the Python dependencies, including PyInstaller, then build on the target OS:

```sh
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm morphos.spec
```

The result is `dist/MORPHOS_Studio.exe` on Windows or `dist/MORPHOS_Studio` on Linux/macOS. Do not copy a Windows executable to Linux/macOS (or vice versa); PyInstaller does not cross-compile.

### Run the services separately

For development or diagnostics, start the daemon and bridge in separate terminals:

```sh
python -m core.gateway.ipc_server --backend morphos --host 127.0.0.1 --port 9876 --data-dir ./data_gateway
python -m core.gateway.web_bridge --host 127.0.0.1 --port 8080 --ipc-host 127.0.0.1 --ipc-port 9876 --data-dir ./data_gateway --dist-dir ./studio/dist
```

The first command starts the full Nengo-backed runtime by default. `--backend aligner` is a lightweight aligner-only mode for gateway development; it does not provide the full MORPHOS neural pipeline.

## Dashboard controls

- **LIVE:** pass incoming samples through without the chaos injector.
- **COMBAT5 INJECT:** enable the configured noise/drop/spike chaos injector for subsequent core input.
- **RESET BREAKER:** reset circuit-breaker state; it does not reset or erase WAL history.
- **Lorenz-63:** start/stop the local numerical source that sends samples to the IPC daemon.
- **Host resources:** start/stop local `psutil` telemetry. These metrics are not injected into the current 3-D neural input.
- **BRANCH SNAPSHOT:** replay a selected retained core state into a counterfactual branch. Supported perturbations are `gain`, `noise`, `step`, and `dropout`.
- **History timeline:** select a WAL record by step. Selecting an older step loads its actual persisted page; **RETURN LIVE** resumes the latest stream.
- **Event log:** initially loads the latest SQLite events; scrolling to the top fetches older pages. Only visible rows are rendered.

## IPC contract

Default endpoint: `127.0.0.1:9876`. Transport: one JSON object per line, one JSON response per request. Requests are validated before they reach the core; malformed JSON, unknown fields, non-finite numbers, invalid dimensions, and invalid controls are rejected.

### Example requests

```json
{"op":"status"}
{"op":"ingest","timestamp":0.005,"values":[1.0,2.0,3.0]}
{"op":"query","timestamp":0.005}
{"op":"control","action":"set_mode","mode":"live"}
{"op":"control","action":"reset_breaker"}
{"op":"control","action":"branch","branch_mode":"gain","step":100}
```

An ingestion vector for the full MORPHOS runtime must have exactly three finite numeric values. Request IDs, when present, are limited to 64 ASCII letters, digits, `.`, `_`, `:`, or `-`. The maximum JSONL request size is 1 MiB.

### HTTP and WebSocket API

Default endpoint: `http://127.0.0.1:8080`.

| Route | Purpose |
|---|---|
| `GET /api/health` | Latest bridge/core status, source states, and connection information |
| `GET /api/history?offset=0&limit=200` | Paged records from the actual binary WAL (`limit` 1–500) |
| `GET /api/events?offset=0&limit=100` | Paged SQLite events, newest page first (`limit` 1–500) |
| `POST /api/control` | Validated source and core-control requests |
| `WS /ws` | Telemetry snapshots pushed every 50 ms (20 Hz) |

Example HTTP control bodies:

```json
{"action":"source","source":"lorenz","enabled":true}
{"action":"source","source":"system","enabled":true}
{"action":"set_mode","mode":"combat5"}
{"action":"reset_breaker"}
{"action":"branch","branch_mode":"dropout","step":100}
```

Invalid control fields receive an HTTP 422 response; unavailable IPC/core operations receive an HTTP 503 response.

## Storage and configuration

The runtime writes:

- `trajectory.wal` — fixed-size append-only signal records.
- `morphos.db` — SQLite run metadata, events, and snapshots; SQLite uses WAL journaling.

The launcher stores data in the platform’s user data directory by default:

- Windows: `%LOCALAPPDATA%\MORPHOS`
- macOS: `~/Library/Application Support/MORPHOS`
- Linux: `$XDG_DATA_HOME/morphos`, or `~/.local/share/morphos` when `XDG_DATA_HOME` is unset

Set `MORPHOS_DATA_DIR` to use another directory. Other useful environment variables:

| Variable | Default | Description |
|---|---|---|
| `MORPHOS_IPC_HOST` | `127.0.0.1` | IPC bind/connect host |
| `MORPHOS_IPC_PORT` | `9876` | IPC TCP port |
| `MORPHOS_WEB_HOST` | `127.0.0.1` | Web bind host; the supervisor only permits loopback |
| `MORPHOS_WEB_PORT` | `8080` | HTTP/WebSocket port |
| `MORPHOS_NEURONS` | `400` | Nengo ensemble neuron count |
| `MORPHOS_LORENZ` | `1` | Set to `0` to disable the default Lorenz source |
| `MORPHOS_OPEN_BROWSER` | `1` | Set to `0` to suppress automatic browser launch |
| `MORPHOS_STUDIO_DIST` | bundled/source `studio/dist` | Override the static frontend directory |

## Tests

Install the development requirements and run the acceptance suites:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests/concurrency_test.py tests/test_web_bridge.py tests/test_studio_launcher.py
```

`concurrency_test.py` includes the 10-client, 5,000-step IPC workload, thread-safety checks, core/WAL integration, restart timeline continuity, and client-disconnect isolation. The bridge and launcher tests exercise real local IPC, HTTP/WebSocket, SQLite/WAL, and (when present) the packaged Windows executable.

## Security and operational notes

- The default IPC and HTTP listeners bind to loopback. Do not expose these unauthenticated control endpoints to an untrusted network.
- The HTTP API can change runtime mode, toggle sources, reset the breaker, and create branches. It has no login or TLS layer; it is intended for local use.
- WAL and SQLite files contain system-generated telemetry and event records. Keep the data directory out of public source control unless the records have been reviewed and intentionally sanitized.
- The repository has no license file at this time. Add a license only after deciding the project’s reuse and redistribution terms.

## Repository map

| Path | Contents |
|---|---|
| `core/gateway/` | Strict IPC schemas, full runtime, TCP daemon, HTTP/WebSocket bridge |
| `core/ingest/` | Time alignment, ring buffer, optional trace ingestion |
| `core/dynamics/` | Nengo simulator, multiscale and source-weaving engines |
| `core/encoding/` | Feature encoding and orthogonal projection |
| `core/prediction/` | Online NLMS prediction |
| `core/observer/` | Dual-opinion anomaly scoring and circuit-breaker logic |
| `core/memory/` | Thread-safe temporal buffers, associative memory, compression |
| `core/counterfactual/` | Snapshot and counterfactual branch engine |
| `adapters/` | Lorenz, host metrics, chaos, SDK, and Rust trace adapters |
| `engines/rust_trace/` | Optional Rust trace reader, writer, and FFI source |
| `cli/`, `tools/` | Nim CLI and trace/export utilities |
| `docs/` | API, SDK, and architecture/technical notes |
| `storage/` | Binary WAL and SQLite persistence |
| `studio/` | React/TypeScript dashboard source and built static assets |
| `tests/` | Core, concurrency, bridge, launcher, SDK, and optional Rust tests |
| `morphos.spec` | PyInstaller one-file application definition |
| `start_morphos.bat`, `start_morphos.sh` | Windows and POSIX launch helpers |

# 中文說明

## 專案簡介

**PROJECT MORPHOS** 是一套本機優先、以 Nengo 為核心的神經動態與遙測平台。它結合 Python 神經計算核心、具嚴格驗證的 TCP/JSONL 閘道器、追加式 WAL 與 SQLite 儲存，以及 React 儀表板。核心守護進程與 Web Bridge 分開執行；瀏覽器透過本機 Bridge 讀取核心遙測與歷史資料。

> **專案狀態：** Windows 單檔執行檔及本機 IPC／HTTP／WebSocket 流程已實際測試。Linux 與 macOS 的啟動腳本及 PyInstaller 設定已提供，但原生執行檔必須在目標作業系統上建置與驗收。
>
> **範圍說明：** 目前使用的儀表板原始碼在 `studio/`。`visualization/` 是較早期、獨立的視覺化／展示程式，不是 MORPHOS Studio 的資料來源，也不由目前 Launcher 託管。
>
> **授權：** 目前專案沒有 `LICENSE` 檔；本說明文件不授予重用、修改或再散布本專案的權利。

## 功能一覽

- 以 5 ms 固定步長處理三維資料，實際執行 Nengo 模擬器 `NengoDynamics`、特徵編碼、線上 NLMS 預測、異常分數與熔斷器。
- 透過 loopback TCP 上的 JSON Lines（JSONL）接收資料及控制請求；請求會先經嚴格 Schema 驗證。
- 每筆軌跡寫入追加式二進位 WAL；執行階段、中繼資料與事件寫入 SQLite WAL 模式資料庫。
- FastAPI HTTP/WebSocket Bridge 預設每秒推送 20 次遙測，提供真實 WAL 和 SQLite 分頁 API。
- React/TypeScript 儀表板提供訊號圖、歷史滑動檢視、虛擬事件日誌、主機指標與核心控制。
- 可將本機 Lorenz-63 數值積分資料送入核心；主機 CPU、記憶體、磁碟與網路指標則作為獨立 Bridge 遙測。
- 包含執行緒安全、用戶端斷線隔離、WAL 重啟、Bridge 與打包啟動器測試。

### 資料真實性與指標解讀

- Studio 的波形和事件來自 Bridge 遙測、二進位 WAL 及 SQLite。沒有資料時顯示空狀態，不會補入合成樣本。
- `combat5` 會對送入核心的資料啟用設定好的 ChaosInjector；要使用未經混沌注入的輸入，請切換 `live`。
- Lorenz 資料由本機 Lorenz-63 數值積分器計算後送入 IPC 核心。
- 主機資源是 `psutil` 實際讀值，但目前只作獨立監控；目前的三維 MORPHOS runtime **不會**把四維主機向量融合進 Nengo 輸入。
- 「Basis cross-talk」是設定之正交投影矩陣間的幾何重疊度，並非兩個資料流共同進入單一神經網路後的端到端串擾量測。
- 反事實分支會回放核心保留在記憶體中的狀態；目前只保留最近 1,024 筆狀態供建立分支，超出範圍的步數會被拒絕。
- 舊的 `visualization/` 程式不是目前 Studio 儀表板的資料來源。

## 架構與資料流

```text
Lorenz-63 資料源 ───────────────┐
外部 JSONL 用戶端 ──────────────┼──> IPC 守護進程（Nengo／預測／異常偵測）
                              │                │
                              │                ├──> trajectory.wal
                              │                └──> morphos.db（SQLite/WAL）
                              │
瀏覽器 <── HTTP/WebSocket <── Web Bridge ──────┴── 讀取歷史與事件
                                │
                                └── 本機 psutil 主機指標
```

啟動器會管理兩個子行程：

1. **IPC 核心守護進程：** 擁有 `MORPHOSRuntime`、Nengo、資料對齊器、預測器、異常判斷與持久化寫入器。
2. **Web Bridge：** 透過 IPC 查詢守護進程、讀取 WAL／SQLite 歷史，並託管 Studio 靜態檔案。

關閉瀏覽器分頁不會停止核心或 Bridge。停止啟動器時會要求兩個子行程正常關閉。

## 系統需求

- 原始碼模式需要 Python 3.10 以上。
- 只有在重新建置 Studio 前端時才需要 Node.js 18 以上。
- 原始碼可在 Windows、Linux、macOS 執行。PyInstaller 產物依作業系統而異，請在執行目標系統上建置。

## 快速開始

### Windows 原始碼模式

```powershell
cd MORPHOS
python -m pip install -r requirements.txt
npm --prefix studio install
npm --prefix studio run build
python studio_launcher.py
```

啟動器會開啟 `http://127.0.0.1:8080`。前端建置完成後，也可雙擊 `start_morphos.bat`。

### Windows 單檔執行檔

依「建置與啟動」章節完成前端及執行檔建置後，雙擊 `start_morphos.bat`，或在 PowerShell 執行：

```powershell
.\dist\MORPHOS_Studio.exe
```

執行檔已包含 Python、Nengo、Nengo 必要資料表、Web Bridge 及編譯後的 Studio 靜態檔。使用單檔版不需要另外安裝 Python 或 Node.js。

### Linux / macOS 原始碼模式

```sh
python3 -m pip install -r requirements.txt
npm --prefix studio install
npm --prefix studio run build
chmod +x start_morphos.sh
./start_morphos.sh
```

瀏覽器網址為 `http://127.0.0.1:8080`。若要產生該作業系統的獨立執行檔，請在該系統上建置 `morphos.spec`。

## 建置與啟動

### 建置前端

```sh
npm --prefix studio install
npm --prefix studio run build
```

產物放在 `studio/dist/`，由 Python Bridge 託管；使用者執行時不需要 Node.js。

### 建置獨立執行檔

安裝 Python 依賴（包含 PyInstaller），並在目標作業系統上執行：

```sh
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm morphos.spec
```

Windows 輸出為 `dist/MORPHOS_Studio.exe`；Linux/macOS 輸出為 `dist/MORPHOS_Studio`。PyInstaller 不提供跨作業系統編譯，請勿把 Windows 執行檔直接拿到 Linux/macOS 使用，反之亦然。

### 分別啟動服務

開發或除錯時，可以在兩個終端機分別執行：

```sh
python -m core.gateway.ipc_server --backend morphos --host 127.0.0.1 --port 9876 --data-dir ./data_gateway
python -m core.gateway.web_bridge --host 127.0.0.1 --port 8080 --ipc-host 127.0.0.1 --ipc-port 9876 --data-dir ./data_gateway --dist-dir ./studio/dist
```

第一條命令預設啟動完整 Nengo runtime。`--backend aligner` 是供閘道器開發使用的輕量對齊器模式，不含完整 MORPHOS 神經處理流程。

## 儀表板控制

- **LIVE：** 後續輸入不經 ChaosInjector。
- **COMBAT5 INJECT：** 對之後輸入啟用設定好的雜訊／丟棄／尖峰混沌注入器。
- **RESET BREAKER：** 重置熔斷器狀態；不會清除 WAL 歷史。
- **Lorenz-63：** 啟動／停止本機數值資料源，並將樣本送到 IPC 核心。
- **Host resources：** 啟動／停止本機 `psutil` 遙測；這些資料不會注入目前的三維神經輸入。
- **BRANCH SNAPSHOT：** 從選取的保留狀態建立反事實分支，支援 `gain`、`noise`、`step`、`dropout` 擾動。
- **History timeline：** 依 step 選取 WAL 紀錄。舊 step 會載入實際持久化資料；按 **RETURN LIVE** 回到最新資料流。
- **Event log：** 首次載入最近的 SQLite 事件；捲動到頂端會繼續載入較舊分頁，畫面只渲染可見資料列。

## IPC 合約

預設位址：`127.0.0.1:9876`。傳輸格式為每行一個 JSON 請求，每個請求回傳一行 JSON。請求在進入核心前會經過驗證；錯誤 JSON、未知欄位、非有限數值、維度錯誤或無效控制會被拒絕。

### 請求範例

```json
{"op":"status"}
{"op":"ingest","timestamp":0.005,"values":[1.0,2.0,3.0]}
{"op":"query","timestamp":0.005}
{"op":"control","action":"set_mode","mode":"live"}
{"op":"control","action":"reset_breaker"}
{"op":"control","action":"branch","branch_mode":"gain","step":100}
```

完整 MORPHOS runtime 的輸入必須剛好是三個有限數值。`request_id` 若提供，最多 64 個 ASCII 字母、數字或 `. _ : -`。JSONL 單筆請求上限為 1 MiB。

### HTTP 與 WebSocket API

預設網址：`http://127.0.0.1:8080`。

| 路由 | 用途 |
|---|---|
| `GET /api/health` | 最新 Bridge／核心狀態、資料源狀態與連線資訊 |
| `GET /api/history?offset=0&limit=200` | 分頁讀取實際二進位 WAL（`limit` 為 1–500） |
| `GET /api/events?offset=0&limit=100` | 分頁讀取 SQLite 事件（最新分頁優先，`limit` 為 1–500） |
| `POST /api/control` | 驗證並轉送資料源或核心控制 |
| `WS /ws` | 每 50 ms 推送一次遙測快照（20 Hz） |

HTTP 控制請求範例：

```json
{"action":"source","source":"lorenz","enabled":true}
{"action":"source","source":"system","enabled":true}
{"action":"set_mode","mode":"combat5"}
{"action":"reset_breaker"}
{"action":"branch","branch_mode":"dropout","step":100}
```

控制欄位錯誤時回傳 HTTP 422；IPC／核心不可用時回傳 HTTP 503。

## 儲存與設定

runtime 會寫入：

- `trajectory.wal`：固定大小、追加式訊號紀錄。
- `morphos.db`：SQLite 執行階段、中繼資料、事件與快照資料庫；使用 SQLite WAL journaling。

啟動器預設將資料放在各平台使用者資料目錄：

- Windows：`%LOCALAPPDATA%\MORPHOS`
- macOS：`~/Library/Application Support/MORPHOS`
- Linux：`$XDG_DATA_HOME/morphos`；未設定 `XDG_DATA_HOME` 時使用 `~/.local/share/morphos`

可設定 `MORPHOS_DATA_DIR` 改用其他目錄。其他常用環境變數：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MORPHOS_IPC_HOST` | `127.0.0.1` | IPC 綁定／連線主機 |
| `MORPHOS_IPC_PORT` | `9876` | IPC TCP 埠號 |
| `MORPHOS_WEB_HOST` | `127.0.0.1` | Web 綁定主機；啟動器僅允許 loopback |
| `MORPHOS_WEB_PORT` | `8080` | HTTP/WebSocket 埠號 |
| `MORPHOS_NEURONS` | `400` | Nengo ensemble 神經元數量 |
| `MORPHOS_LORENZ` | `1` | 設為 `0` 可關閉預設 Lorenz 資料源 |
| `MORPHOS_OPEN_BROWSER` | `1` | 設為 `0` 可停用自動開啟瀏覽器 |
| `MORPHOS_STUDIO_DIST` | 打包／原始碼中的 `studio/dist` | 指定靜態前端目錄 |

## 測試

安裝開發依賴並執行主要驗收套件：

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests/concurrency_test.py tests/test_web_bridge.py tests/test_studio_launcher.py
```

`concurrency_test.py` 包含 10 客戶端、5,000 步 IPC 壓測、執行緒安全、核心／WAL 接線、重啟後時間軸連續性及用戶端斷線隔離。Bridge 與 Launcher 測試會實際啟動本機 IPC、HTTP/WebSocket、SQLite/WAL；若 Windows 執行檔存在，也會測試打包版。

## 安全與操作注意事項

- IPC 與 HTTP 預設只監聽 loopback。不要將沒有身份驗證的控制 API 暴露給不可信任網路。
- HTTP API 可以切換 runtime 模式、資料源、重置熔斷器及建立分支。目前沒有登入或 TLS，設計用途是本機操作。
- WAL 與 SQLite 會記錄遙測和事件。除非已檢查並刻意清理內容，請勿將資料目錄加入公開版控。
- 目前沒有專案授權檔。請先決定重用與再散布條款，再自行加入授權。

## 主要目錄

| 路徑 | 內容 |
|---|---|
| `core/gateway/` | 嚴格 IPC Schema、完整 runtime、TCP 守護進程、HTTP/WebSocket Bridge |
| `core/ingest/` | 時間對齊、環形緩衝、選用軌跡資料接入 |
| `core/dynamics/` | Nengo 模擬器、多尺度與異質來源編織引擎 |
| `core/encoding/` | 特徵編碼與正交投影 |
| `core/prediction/` | 線上 NLMS 預測 |
| `core/observer/` | Dual-Opinion 異常分數與熔斷器 |
| `core/memory/` | 執行緒安全時序緩衝、關聯記憶與壓縮 |
| `core/counterfactual/` | 快照與反事實分支引擎 |
| `adapters/` | Lorenz、主機指標、混沌、SDK 與 Rust 軌跡資料源 |
| `engines/rust_trace/` | 選用 Rust 軌跡讀寫器及 FFI 原始碼 |
| `cli/`、`tools/` | Nim 命令列工具與軌跡／匯出工具 |
| `docs/` | API、SDK 與技術說明文件 |
| `storage/` | 二進位 WAL 與 SQLite 持久化 |
| `studio/` | React/TypeScript 儀表板原始碼與靜態建置檔 |
| `tests/` | 核心、併發、Bridge、啟動器、SDK 與選用 Rust 測試 |
| `morphos.spec` | PyInstaller 單檔應用程式設定 |
| `start_morphos.bat`、`start_morphos.sh` | Windows 與 POSIX 啟動腳本 |
