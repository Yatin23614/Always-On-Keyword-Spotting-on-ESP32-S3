Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($r in @(-2, 0, 1, 2, 3, 4)) {
    $synth.Rate = $r
    $file = "test_rate_$r.wav"
    $synth.SetOutputToWaveFile($file, $format)
    $synth.Speak("hello jarvis development mode on")
    $synth.SetOutputToNull()
    $len = (Get-Item $file).Length
    $dur = ($len - 44) / 32000.0
    Write-Host ("Rate {0,2}: {1:F2} seconds ({2} samples)" -f $r, $dur, (($len - 44)/2))
    Remove-Item $file
}
$synth.Dispose()
