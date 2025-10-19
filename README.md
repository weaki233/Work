xgboost_shap 打包的关键：
1. 针对打包后的卡死问题（出问题时先用console模式查看问题）
  ```python
  # py文件
  # --- 强制标准输出/错误流使用 UTF-8 编码并启用行缓冲 ---
  # 这是一个处理打包后程序（尤其是在Windows上）Unicode错误和输出延迟问题的稳定方法。
  # line_buffering=True 确保每行 print 输出后都会立即刷新，实现实时显示。
  if sys.stdout is not None:
      sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
  if sys.stderr is not None:
      sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
  # --- 修复结束 ---
  ```
2. 针对Xgboost打包不全的问题（先用pyinstaller打包后，修改其生成的.spec文件）
  ```python
# .spec中新增：
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
  ```
