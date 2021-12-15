# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None


hiddenimports = collect_submodules('tensorflow')
hiddenimports.extend(collect_submodules('dgl'))
print(hiddenimports)

datas = collect_data_files('tensorflow', subdir=None, include_py_files=True)
datas2 = collect_data_files('dgl', subdir=None, include_py_files=True)

hiddenimports.append('tensorflow._api.v2.compat.v1.compat.v2.keras.callbacks')

added_files = [('templates', 'templates'), ('static', 'static'),('MOA.db','.'),('requirements.txt','.'),('Dataset','Dataset')]
#added_files.append(("jre8/win","jre8/win"))
added_files.extend(datas)
added_files.extend(datas2)

a = Analysis(['run.py'],
             pathex=['.','MOA_integration/Models'],
             binaries=[],
             datas= added_files,
             hiddenimports = hiddenimports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

#to_remove = ['_pywrap_tensorflow_interpreter_wrapper','_pywrap_tensorflow_lite_calibration_wrapper','_dtypes','_errors_test_helper','_op_def_registry','_op_def_util','_proto_comparators','_python_memory_checker_helper','_pywrap_debug_events_writer','_pywrap_device_lib','_pywrap_events_writer','_pywrap_mlir','_pywrap_parallel_device','_pywrap_py_exception_registry','_pywrap_python_api_dispatcher','_pywrap_python_api_info']
#for b in a.binaries:
#    found = any(f'{crypto}.pyd' in b[1]
#                for crypto in to_remove
#                )
#    if found:
#        print(f"Removing {b[1]}")
#        a.binaries.remove(b)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)



exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='run',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
