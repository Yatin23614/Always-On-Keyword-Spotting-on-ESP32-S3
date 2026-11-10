Add-Type -AssemblyName System.Speech

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.SetOutputToWaveFile("test_phrase.wav", $format)
$synth.Speak("hello jarvis development mode on")
$synth.Dispose()

$len = (Get-Item "test_phrase.wav").Length
$dur = ($len - 44) / 32000.0
Write-Host "Duration: $dur s, Length: $len bytes"

