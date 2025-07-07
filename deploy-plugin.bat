@echo off
setlocal

echo Deploying plugin to Program Files...

REM タイマースタート
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a start_seconds=%%a*3600 + %%b*60 + %%c
)

REM Blenderアドオンの配置パス（Program Files 側）
set "BLENDER_ADDONS=%ProgramFiles%\Blender Foundation\Blender 3.4\3.4\scripts\addons"
set "PLUGIN_NAME=plugin"
set "PLUGIN_SRC=plugin"

REM 古いプラグイン削除
echo Deleting old plugin from: "%BLENDER_ADDONS%\%PLUGIN_NAME%"
rmdir /s /q "%BLENDER_ADDONS%\%PLUGIN_NAME%"

REM 新しいプラグインコピー（\*付きで中身のみコピー）
echo Copying new plugin from "%PLUGIN_SRC%" to "%BLENDER_ADDONS%\%PLUGIN_NAME%"
mkdir "%BLENDER_ADDONS%\%PLUGIN_NAME%"
xcopy /E /I /Y "%PLUGIN_SRC%\*" "%BLENDER_ADDONS%\%PLUGIN_NAME%"

REM Blender 非GUIモードでプラグイン展開スクリプト実行
set "BLENDER_EXE=%ProgramFiles%\Blender Foundation\Blender 3.4\blender.exe"
"%BLENDER_EXE%" -b -P "%BLENDER_ADDONS%\%PLUGIN_NAME%\deploy-plugin.py"

REM タイマー終了
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a end_seconds=%%a*3600 + %%b*60 + %%c
)
set /a runtime=%end_seconds% - %start_seconds%

echo.
echo Deployed successfully in %runtime% second(s)!
endlocal
