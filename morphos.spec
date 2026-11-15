# Build on the target operating system; PyInstaller binaries are platform-specific.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH)
studio_dist = root / "studio" / "dist"
if not (studio_dist / "index.html").is_file():
    raise FileNotFoundError("Build studio/dist before running PyInstaller")

a = Analysis(
    [str(root / "studio_launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=collect_data_files("nengo") + [(str(studio_dist), "studio/dist")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MORPHOS_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)
