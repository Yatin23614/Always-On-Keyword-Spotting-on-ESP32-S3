param(
    [string]$OutputDir = "data/raw/unknown_phrases"
)

Add-Type -AssemblyName System.Speech

$voices = @("Microsoft David Desktop", "Microsoft Zira Desktop")
$negative_phrases = @(
    "hello computer",
    "hey siri turn on the light",
    "alexa play music",
    "okay google what is the weather",
    "jarvis can you hear me",
    "turn on development mode",
    "switch mode on",
    "hello how are you",
    "system status check",
    "open the browser window",
    "start the development server",
    "stop the machine",
    "turn off the lights please",
    "increase the volume to maximum",
    "good morning jarvis",
    "what time is it right now"
)

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
        foreach ($phrase in $negative_phrases) {
            foreach ($rate in @(-1, 0, 1, 2)) {
                $count++
                $synth.Rate = $rate
                $synth.Volume = 90
                $filename = ("neg_{0:D4}.wav" -f $count)
                $path = Join-Path $OutputDir $filename
                $synth.SetOutputToWaveFile($path, $format)
                $synth.Speak($phrase)
                $synth.SetOutputToNull()
            }
        }
    }
    Write-Host ("Generated {0} negative speech samples in {1}" -f $count, $OutputDir)
}
finally {
    $synth.Dispose()
}

