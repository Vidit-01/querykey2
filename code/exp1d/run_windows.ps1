$ErrorActionPreference = "Stop"
$Preset = if ($args.Count -ge 1) { $args[0] } else { "quick" }
$ShardIndex = if ($args.Count -ge 2) { $args[1] } else { "0" }
$NumShards = if ($args.Count -ge 3) { $args[2] } else { "1" }
$DataFile = if ($args.Count -ge 4) { $args[3] } else { "" }
$Extra = @()
if ($DataFile) {
  $Extra += @("--tinystories-file", $DataFile)
}
if (-not (Test-Path "code/exp1d/data/$Preset/selected_points.json")) {
  python code/exp1d/run.py --preset $Preset --mode select
}
python code/exp1d/run.py `
  --preset $Preset `
  --mode run `
  --shard-index $ShardIndex `
  --num-shards $NumShards `
  @Extra
if ($NumShards -eq "1") {
  python code/exp1d/run.py --preset $Preset --mode analyze
}
