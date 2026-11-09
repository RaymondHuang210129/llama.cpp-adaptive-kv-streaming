param([Parameter(Mandatory = $true)][int]$RootPid)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ids = [System.Collections.Generic.HashSet[int]]::new()
$null = $ids.Add($RootPid)
$tree = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
do {
    $changed = $false
    foreach ($item in $tree) {
        if ($ids.Contains([int]$item.ParentProcessId) -and $ids.Add([int]$item.ProcessId)) {
            $changed = $true
        }
    }
} while ($changed)

$processes = @(foreach ($process in (Get-Process -Id @($ids) -ErrorAction SilentlyContinue)) {
    try {
        [ordered]@{
            pid = $process.Id
            name = $process.ProcessName
            cpu_seconds = $process.TotalProcessorTime.TotalSeconds
            working_set_bytes = $process.WorkingSet64
            private_commit_bytes = $process.PrivateMemorySize64
        }
    } catch [System.InvalidOperationException] {
        # A compiler child can exit between enumeration and sampling.
        continue
    }
})
$paths = @(
    '\Memory\Available MBytes',
    '\Memory\Committed Bytes',
    '\Memory\Commit Limit',
    '\Memory\Page Reads/sec',
    '\Memory\Page Writes/sec',
    '\Paging File(_Total)\% Usage',
    '\Processor(_Total)\% Processor Time'
)
$counterErrors = @()
$samples = Get-Counter -Counter $paths -ErrorAction SilentlyContinue -ErrorVariable counterErrors
$cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
[ordered]@{
    processes = $processes
    cpu = $cpu
    counters = @($samples.CounterSamples | Select-Object Path, CookedValue, Status)
    counter_errors = @($counterErrors | ForEach-Object { $_.ToString() })
    pagefiles = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage)
    note = 'Private committed bytes are not a measurement of bytes swapped to disk; paging counters are system-wide.'
} | ConvertTo-Json -Depth 6 -Compress
