#!/bin/bash
# Run GFN2-xTB and g-xTB single-point calculations on conformer ensembles
# Usage: bash run_benchmark.sh <folder> <charge> <njobs>

set -euo pipefail

FOLDER=${1:?"Usage: $0 <folder> <charge> <njobs>"}
CHARGE=${2:?"Usage: $0 <folder> <charge> <njobs>"}
NJOBS=${3:-8}

XTB_GFN2=/home/xchen/software/xtb-6.7.1/xtb-dist/bin/xtb
XTB_GXTB=/home/xchen/software/xtb-6.7.1-gxtb/bin/xtb

OUTDIR="${FOLDER}/xtb_results"
mkdir -p "$OUTDIR"

run_one() {
    local xyz=$1
    local base
    base=$(basename "$xyz" .xyz)
    local wdir="$OUTDIR/${base}"
    mkdir -p "$wdir"

    # GFN2 single-point
    if [[ ! -f "$wdir/gfn2.energy" ]]; then
        $XTB_GFN2 "$xyz" --charge "$CHARGE" --gfn 2 --norestart \
            > "$wdir/gfn2.out" 2>&1 || true
        grep "TOTAL ENERGY" "$wdir/gfn2.out" | tail -1 | awk '{print $4}' \
            > "$wdir/gfn2.energy"
    fi

    # g-xTB single-point
    if [[ ! -f "$wdir/gxtb.energy" ]]; then
        $XTB_GXTB "$xyz" --charge "$CHARGE" --gxtb --norestart \
            > "$wdir/gxtb.out" 2>&1 || true
        grep "TOTAL ENERGY" "$wdir/gxtb.out" | tail -1 | awk '{print $4}' \
            > "$wdir/gxtb.energy"
    fi
}

export -f run_one
export OUTDIR CHARGE XTB_GFN2 XTB_GXTB

find "$FOLDER" -maxdepth 1 -name "snap_*.xyz" | sort | \
    xargs -P "$NJOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "Done: $FOLDER"
