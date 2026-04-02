# build_latex.ps1
# Called by LaTeX Workshop on every save.
# Creates a timestamped subfolder under $OutBase, runs the full
# pdflatex → bibtex → pdflatex × 2 sequence, then copies the PDF +
# SyncTeX to a stable location so the VS Code viewer can find them.
#
# Usage (from LaTeX Workshop tool args):
#   powershell.exe -NonInteractive -NoProfile -ExecutionPolicy Bypass
#     -File build_latex.ps1 -DocPath "%DOC%.tex" -OutBase "%OUTDIR%"

param(
    [string]$DocPath,   # full path to .tex file  (passed as %DOC%.tex)
    [string]$OutBase    # stable output dir: .../out/<docfile>/
)

$miktexBin = "C:\Users\GAK\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
$pdflatex  = "$miktexBin\pdflatex.exe"
$bibtex    = "$miktexBin\bibtex.exe"
$docFile   = [System.IO.Path]::GetFileNameWithoutExtension($DocPath)

# Create timestamped subfolder
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tsDir     = Join-Path $OutBase $timestamp

New-Item -ItemType Directory -Path $OutBase -Force | Out-Null
New-Item -ItemType Directory -Path $tsDir   -Force | Out-Null

$pdfArgs = @(
    "-synctex=1",
    "-interaction=nonstopmode",
    "-file-line-error",
    "-output-directory=$tsDir",
    "-include-directory=$tsDir",
    $DocPath
)

# Pass 1 — generate .aux (needed by bibtex)
& $pdflatex @pdfArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# BibTeX — resolve \cite{} references (only if .aux exists)
$auxFile = Join-Path $tsDir "$docFile"
$srcDir  = [System.IO.Path]::GetDirectoryName($DocPath)
if (Test-Path "$tsDir\$docFile.aux") {
    & $bibtex $auxFile
    # bibtex returns non-zero for warnings (missing entries etc.) — don't abort
}

# Copy .bbl next to the .tex so pdflatex can find it
$bblSrc = Join-Path $tsDir "$docFile.bbl"
if (Test-Path $bblSrc) {
    Copy-Item $bblSrc -Destination (Join-Path $srcDir "$docFile.bbl") -Force
}

# Pass 2 — incorporate bibliography
& $pdflatex @pdfArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Pass 3 — resolve all cross-references
& $pdflatex @pdfArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Copy PDF + SyncTeX to the stable path so the viewer finds them
foreach ($ext in @("pdf", "synctex.gz")) {
    $src = Join-Path $tsDir "$docFile.$ext"
    if (Test-Path $src) {
        Copy-Item $src -Destination (Join-Path $OutBase "$docFile.$ext") -Force
    }
}

Write-Host "Build saved to: $tsDir"
exit 0
