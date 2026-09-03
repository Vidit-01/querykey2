$ErrorActionPreference = "Stop"
$Preset = if ($args.Count -ge 1) { $args[0] } else { "quick" }
$ShardIndex = if ($args.Count -ge 2) { $args[1] } else { "0" }
$NumShards = if ($args.Count -ge 3) { $args[2] } else { "1" }
python code/exp1b/run.py `
  --preset $Preset `
  --mode run `
  --shard-index $ShardIndex `
  --num-shards $NumShards
if ($NumShards -eq "1") {
  python code/exp1b/run.py --preset $Preset --mode analyze
}
