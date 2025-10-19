# -*- mode: python ; coding: utf-8 -*-

# 找到你Python环境中xgboost库的路径
import sys
import os
site_packages_path = next(p for p in sys.path if 'site-packages' in p)
xgboost_path = os.path.join(site_packages_path, 'xgboost')
a = Analysis(
    ['xgboost_shap.py'],
    pathex=[],
    binaries=[],
    datas=[(xgboost_path, 'xgboost')], # <-- 关键！将此行添加到这里
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
    name='SHAP分析工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
