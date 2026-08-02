param(
    [string]$Python = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$entrypoint = Join-Path $projectDir "run_factor_demo.py"

& $Python $entrypoint `
    --debug `
    --factors return_1d `
    @ExtraArgs

exit $LASTEXITCODE
