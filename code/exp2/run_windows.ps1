$ErrorActionPreference = "Stop"
$Preset = if ($args.Count -ge 1) { $args[0] } else { "quick" }
$Study = if ($args.Count -ge 2) { $args[1] } else { "pilot" }
$ShardIndex = if ($args.Count -ge 3) { $args[2] } else { "0" }
$NumShards = if ($args.Count -ge 4) { $args[3] } else { "1" }
$TrainFile = if ($args.Count -ge 5) { $args[4] } else { "" }
$ValidationFile = if ($args.Count -ge 6) { $args[5] } else { "" }
$Extra = @()
if ($TrainFile) {
  $Extra += @("--train-file", $TrainFile)
}
if ($ValidationFile) {
  $Extra += @("--validation-file", $ValidationFile)
}
if (-not (Test-Path "code/exp2/data/$Preset/frozen_treatments.json")) {
  python code/exp2/run.py --preset $Preset --study $Study --mode freeze
}
if (-not (Test-Path "code/exp2/data/$Preset/integration_check.json")) {
  python code/exp2/run.py --preset $Preset --study $Study --mode integration
}
python code/exp2/run.py `
  --preset $Preset `
  --study $Study `
  --mode run `
  --shard-index $ShardIndex `
  --num-shards $NumShards `
  @Extra
if ($NumShards -eq "1") {
  python code/exp2/run.py --preset $Preset --study $Study --mode analyze
}
