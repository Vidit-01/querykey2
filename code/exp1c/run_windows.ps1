$ErrorActionPreference = "Stop"
$Preset = if ($args.Count -ge 1) { $args[0] } else { "quick" }
$ShardIndex = if ($args.Count -ge 2) { $args[1] } else { "0" }
$NumShards = if ($args.Count -ge 3) { $args[2] } else { "1" }
$Points = "code/exp1c/data/$Preset/adaptive_points.json"
if (-not (Test-Path $Points)) {
  python code/exp1c/run.py --preset $Preset --mode propose
}
$Extra = @()
# Pass --replicate-discovery-grid explicitly for preregistered full replication.
python code/exp1c/run.py `
  --preset $Preset `
  --mode run `
  --shard-index $ShardIndex `
  --num-shards $NumShards `
  @Extra
if ($NumShards -eq "1") {
  python code/exp1c/run.py --preset $Preset --mode analyze
}
