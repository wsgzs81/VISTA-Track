param(
    [Parameter(Mandatory=$true)][string]$SeqDir,
    [Parameter(Mandatory=$true)][string]$OutPath,
    [string]$Frames = "0,30,59",
    [ValidateSet("amodal","visible","both")][string]$BoxMode = "amodal"
)

Add-Type -AssemblyName System.Drawing

$FrameList = $Frames -split "," | ForEach-Object { [int]$_.Trim() }

function Read-JsonFile([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Draw-ClippedBox($Graphics, $Pen, [float]$CellX, [float]$CellY, [float]$CellW, [float]$CellH, $Box, [float]$ScaleX, [float]$ScaleY) {
    if ($Box.Count -ne 4) { return }
    $x0 = [Math]::Max(0.0, [double]$Box[0])
    $y0 = [Math]::Max(0.0, [double]$Box[1])
    $x1 = [Math]::Min(1280.0, [double]$Box[0] + [double]$Box[2])
    $y1 = [Math]::Min(720.0, [double]$Box[1] + [double]$Box[3])
    $w = $x1 - $x0
    $h = $y1 - $y0
    if ($w -le 1 -or $h -le 1) { return }
    $Graphics.DrawRectangle($Pen,
        [float]($CellX + $x0 * $ScaleX),
        [float]($CellY + $y0 * $ScaleY),
        [float]($w * $ScaleX),
        [float]($h * $ScaleY))
}

$seq = Get-Item -LiteralPath $SeqDir
$seqMetaPath = Join-Path $seq.FullName "seq_meta.json"
$seqMeta = $null
if (Test-Path -LiteralPath $seqMetaPath) {
    $seqMeta = Read-JsonFile $seqMetaPath
}
$cams = Get-ChildItem -LiteralPath (Join-Path $seq.FullName "frames") -Directory |
    Where-Object { $_.Name -like "cam_*" } |
    Sort-Object Name

if ($cams.Count -eq 0) {
    throw "No camera folders found under $SeqDir"
}

$thumbW = 360
$thumbH = 203
$labelH = 28
$cols = $FrameList.Count
$rows = $cams.Count
$canvasW = $cols * $thumbW
$canvasH = $rows * ($thumbH + $labelH)

$bmp = New-Object System.Drawing.Bitmap $canvasW, $canvasH
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::FromArgb(24, 24, 24))
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$font = New-Object System.Drawing.Font("Consolas", 10)
$brush = [System.Drawing.Brushes]::White
$labelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(220, 0, 0, 0))
$penAmodal = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80, 255, 120), 3)
$penVisible = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(230, 255, 180, 30), 2)

for ($r = 0; $r -lt $rows; $r++) {
    $cam = $cams[$r]
    for ($c = 0; $c -lt $cols; $c++) {
        $frame = $FrameList[$c]
        $frameName = "{0:D6}.png" -f $frame
        $annName = "{0:D6}.json" -f $frame
        $rgbPath = Join-Path $cam.FullName "rgb\$frameName"
        $annPath = Join-Path $cam.FullName "ann\$annName"
        $x = $c * $thumbW
        $y = $r * ($thumbH + $labelH)

        if (Test-Path -LiteralPath $rgbPath) {
            $img = [System.Drawing.Image]::FromFile($rgbPath)
            $g.DrawImage($img, $x, $y + $labelH, $thumbW, $thumbH)

            if (Test-Path -LiteralPath $annPath) {
                $ann = Read-JsonFile $annPath
                $sx = $thumbW / $img.Width
                $sy = $thumbH / $img.Height
                $vis = $ann.bbox_2d_visible_xywh
                $am = $ann.bbox_2d_amodal_xywh
                $occ = $ann.visibility.occlusion_state
                $target = if ($ann.category) { $ann.category } elseif ($seqMeta -and $seqMeta.target_category) { $seqMeta.target_category } else { "unknown" }

                if ($BoxMode -eq "amodal" -or $BoxMode -eq "both") {
                    Draw-ClippedBox $g $penAmodal $x ($y + $labelH) $thumbW $thumbH $am $sx $sy
                }
                if ($BoxMode -eq "visible" -or $BoxMode -eq "both") {
                    Draw-ClippedBox $g $penVisible $x ($y + $labelH) $thumbW $thumbH $vis $sx $sy
                }
                $label = "{0} f{1} target={2} box={3} {4} vis={5:N2}" -f $cam.Name, $frame, $target, $BoxMode, $occ, [double]$ann.visibility.visibility_ratio
            } else {
                $label = "{0} f{1} missing ann" -f $cam.Name, $frame
            }
            $img.Dispose()
        } else {
            $label = "{0} f{1} missing rgb" -f $cam.Name, $frame
        }

        $g.FillRectangle($labelBrush, $x, $y, $thumbW, $labelH)
        $g.DrawString($label, $font, $brush, $x + 6, $y + 7)
    }
}

$outDir = Split-Path -Parent $OutPath
if ($outDir) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$bmp.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)

$labelBrush.Dispose()
$penVisible.Dispose()
$penAmodal.Dispose()
$font.Dispose()
$g.Dispose()
$bmp.Dispose()
Write-Host $OutPath
