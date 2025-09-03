#!/usr/bin/env bash
# Orchestrator for smoke/reproduce/figures with error trapping and bounded steps
set -euo pipefail
trap 'st=$?; echo "ERROR: command failed on line ${BASH_LINENO[0]}"; exit $st' ERR

cmd=${1:-help}
case "$cmd" in
  smoke)
    make setup
    make smoke
    ;;
  reproduce)
    make setup
    make smoke
    make reproduce
    ;;
  figures)
    make figures
    ;;
  *)
    echo "Usage: $0 {smoke|reproduce|figures}";
    ;;
 esac

