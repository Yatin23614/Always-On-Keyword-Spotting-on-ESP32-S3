param(
    [string]$OutputDir = "data/raw/hello_jarvis"
)

Add-Type -AssemblyName System.Speech

$voices = @("Microsoft David Desktop", "Microsoft Zira Desktop")
$phrases = @(
    "hello jarvis development mode on",
    "hello jarvis devlopment mode on",
    "hello jarvis, development mode on",
    "hello jarvis dev mode on",
    "jarvis development mode on"
)
$rates = @(-3, -2, -1, 0, 1, 2, 3, 4)
$volumes = @(80, 90, 100)

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$count = 0
try {
    foreach ($v in $voices) {
        $synth.SelectVoice($v)
        foreach ($phrase in $phrases) {
            foreach ($rate in $rates) {
                foreach ($vol in $volumes) {
                    $count++
                    $synth.Rate = $rate
                    $synth.Volume = $vol
                    $filename = ("jarvis_{0:D4}.wav" -f $count)
                    $path = Join-Path $OutputDir $filename
                    $synth.SetOutputToWaveFile($path, $format)
                    $synth.Speak($phrase)
                    $synth.SetOutputToNull()
                }
            }
        }
    }
    Write-Host ("Generated {0} samples of Hello Jarvis in {1}" -f $count, $OutputDir)
}
finally {
    $synth.Dispose()
}
