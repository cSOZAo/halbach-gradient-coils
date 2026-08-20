<#
.SYNOPSIS
    Empaqueta pyCoilGen-0.2.4 para traslado a otro computador.

.DESCRIPTION
    Crea un ZIP de esta carpeta base (ubicación del script), excluyendo
    metadatos y cachés locales (.git, .cursor, __pycache__, *.pyc).
    Conserva resultados, logs, .gitignore y código obsoleto.

    El ZIP se escribe junto a la carpeta base (no dentro de ella) para
    evitar incluir el archivo dentro de sí mismo.
#>

[CmdletBinding()]
param(
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"

$BaseDir = $PSScriptRoot
if (-not $BaseDir) {
    $BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$BaseDir = (Resolve-Path -LiteralPath $BaseDir).Path
$BaseName = Split-Path -Leaf $BaseDir
$ParentDir = Split-Path -Parent $BaseDir

if (-not $OutputZip) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputZip = Join-Path $ParentDir ("{0}_transfer_{1}.zip" -f $BaseName, $stamp)
} else {
    $OutputZip = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputZip)
}

$ExcludeDirNames = @(".git", ".cursor", "__pycache__")
$ExcludeFileExtensions = @(".pyc")

Write-Host "Carpeta base : $BaseDir"
Write-Host "ZIP destino  : $OutputZip"

if (Test-Path -LiteralPath $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipStream = [System.IO.File]::Open($OutputZip, [System.IO.FileMode]::CreateNew)
$zip = New-Object System.IO.Compression.ZipArchive($zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
$filesAdded = 0
$bytesAdded = [int64]0

try {
    $files = Get-ChildItem -LiteralPath $BaseDir -Recurse -File -Force
    foreach ($file in $files) {
        $full = $file.FullName

        # Never pack the destination ZIP if it somehow lives under BaseDir.
        if ($full -eq $OutputZip) { continue }

        $rel = $full.Substring($BaseDir.Length).TrimStart("\", "/")
        $parts = $rel -split "[\\/]"

        $skip = $false
        foreach ($part in $parts) {
            if ($ExcludeDirNames -contains $part) {
                $skip = $true
                break
            }
        }
        if ($skip) { continue }

        if ($ExcludeFileExtensions -contains $file.Extension.ToLowerInvariant()) {
            continue
        }

        $entryName = ($BaseName + "/" + ($rel -replace "\\", "/"))
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $full, $entryName, [System.IO.Compression.CompressionLevel]::Optimal
        )
        $filesAdded++
        $bytesAdded += $file.Length
    }
}
finally {
    $zip.Dispose()
    $zipStream.Dispose()
}

$zipInfo = Get-Item -LiteralPath $OutputZip
Write-Host ""
Write-Host "Listo."
Write-Host ("  Archivos incluidos : {0}" -f $filesAdded)
Write-Host ("  Bytes fuente       : {0:N0}" -f $bytesAdded)
Write-Host ("  Tamaño ZIP         : {0:N2} MB" -f ($zipInfo.Length / 1MB))
Write-Host ("  Ruta               : {0}" -f $zipInfo.FullName)
