$ErrorActionPreference = "SilentlyContinue"

$command = Get-Command cmake.exe | Select-Object -First 1
if ($command -and $command.Source) {
    [Console]::Out.Write($command.Source.Replace('\', '/'))
    exit 0
}

$candidates = @(
    "$env:ProgramFiles\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Preview\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

$found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($found) {
    [Console]::Out.Write($found.Replace('\', '/'))
}
