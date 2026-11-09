param(
    [string]$OutputDir = "data/raw",
    [int]$CopiesPerLabel = 25
)

Add-Type -AssemblyName System.Speech

$labels = @{
    "start" = "start"
    "stop" = "stop"
    "next" = "next"
    "previous" = "previous"
    "volume_up" = "volume up"
    "volume_down" = "volume down"
    "lights_on" = "lights on"
    "lights_off" = "lights off"
}

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Volume = 100

try {
    foreach ($label in $labels.Keys) {
        $directory = Join-Path $OutputDir $label
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        for ($index = 1; $index -le $CopiesPerLabel; $index++) {
            $rate = (($index - 1) % 5) - 2
            $synth.Rate = $rate
            $path = Join-Path $directory ("tts_{0:D3}.wav" -f $index)
            $synth.SetOutputToWaveFile($path, $format)
            $synth.Speak($labels[$label])
            $synth.SetOutputToNull()
        }
        Write-Host ("Generated {0} files in {1}" -f $CopiesPerLabel, $directory)
    }
}
finally {
    $synth.Dispose()
}
