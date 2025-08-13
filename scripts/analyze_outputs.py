#!/usr/bin/env python
#
# ROBUST TRACE ANALYZER
# - Reads all outputs/*.json files
# - Handles missing data gracefully using .get()
# - Generates a clear Markdown evaluation report
#
import sys, json, glob, os, statistics as stats
from collections import Counter, defaultdict

def load_json(path):
    """Safely loads a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_traces(obj):
    """Analyzes a single trace object and returns a summary dict."""
    traces = obj.get("traces", [])
    n = len(traces)
    if n == 0:
        return {"warning": "no traces found in JSON file"}

    # Safely extract metrics, defaulting to 0
    finals = [t.get("final_accuracy", 0) for t in traces]
    turns_per_item = [len(t.get("turns", [])) for t in traces]
    first_acc = [t["turns"][0].get("accuracy", 0) if t.get("turns") else 0 for t in traces]
    delta_acc = [fin - fst for fin, fst in zip(finals, first_acc)]

    bias_counts = Counter()
    bias_deltas = defaultdict(list)
    template_improve = defaultdict(lambda: [0, 0])  # [improved, total]

    for t in traces:
        turns = t.get("turns", [])
        if not turns:
            continue
        # Per-bias metrics
        last_turn = turns[-1]
        bias = last_turn.get("teacher_bias", "None") or "None"
        bias_counts[bias] += 1
        bias_deltas[bias].append(t.get("final_accuracy", 0) - turns[0].get("accuracy", 0))

        # Per-template metrics
        for i in range(1, len(turns)):
            tmpl = turns[i].get("template") or "None"
            if tmpl != "None":
                prev_acc = turns[i - 1].get("accuracy", 0)
                cur_acc = turns[i].get("accuracy", 0)
                template_improve[tmpl][1] += 1
                if cur_acc > prev_acc:
                    template_improve[tmpl][0] += 1

    def calc_rate(pair):
        improved, total = pair
        return improved / total if total > 0 else 0.0

    summary = {
        "items": n,
        "final_accuracy_mean": stats.mean(finals) if finals else 0.0,
        "turns_mean": stats.mean(turns_per_item) if turns_per_item else 0.0,
        "delta_accuracy_mean": stats.mean(delta_acc) if delta_acc else 0.0,
        "improved_items": sum(1 for d in delta_acc if d > 0),
        "worsened_items": sum(1 for d in delta_acc if d < 0),
        "unchanged_items": sum(1 for d in delta_acc if d == 0),
    }

    per_bias = {
        b: {
            "count": count,
            "delta_accuracy_mean": stats.mean(bias_deltas[b]) if bias_deltas[b] else 0.0,
        } for b, count in sorted(bias_counts.items())
    }

    per_template = {
        t: {
            "uses": pair[1],
            "next_turn_improve_rate": calc_rate(pair),
        } for t, pair in sorted(template_improve.items())
    }

    return {"summary": summary, "per_bias": per_bias, "per_template": per_template}


def main(paths):
    """Main function to generate reports from JSON files."""
    if not paths:
        paths = sorted(glob.glob("outputs/*.json"))
    if not paths:
        print("No outputs/*.json found to analyze.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(paths)} JSON file(s)...")
    reports = {}
    for p in paths:
        try:
            obj = load_json(p)
            reports[os.path.basename(p)] = analyze_traces(obj)
        except Exception as e:
            reports[os.path.basename(p)] = {"error": f"Failed to process: {e}"}

    # Write Markdown Report
    os.makedirs("outputs", exist_ok=True)
    md_path = "outputs/eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Teacher/Learner Evaluation Report\n\n")
        for fname, rep in reports.items():
            f.write(f"## Analysis for `{fname}`\n\n")
            if "error" in rep:
                f.write(f"❗️ **Error:** {rep['error']}\n\n---\n\n")
                continue
            if "warning" in rep:
                f.write(f"⚠️ **Warning:** {rep['warning']}\n\n---\n\n")
                continue

            s = rep["summary"]
            f.write("**Summary**\n\n")
            f.write(f"- Items Processed: {s['items']}\n")
            f.write(f"- Final Accuracy (Mean): {s['final_accuracy_mean']:.3f}\n")
            f.write(f"- Turns per Item (Mean): {s['turns_mean']:.2f}\n")
            f.write(f"- Accuracy Change (Mean Delta): {s['delta_accuracy_mean']:.3f}\n")
            f.write(f"- Item Trajectory (Improved / Worsened / Unchanged): {s['improved_items']} / {s['worsened_items']} / {s['unchanged_items']}\n\n")

            if rep.get("per_bias"):
                f.write("**Per-Bias Analysis**\n\n")
                for b, v in rep["per_bias"].items():
                    f.write(f"- **{b}**: Count={v['count']}, Mean ΔAccuracy={v['delta_accuracy_mean']:.3f}\n")
                f.write("\n")

            if rep.get("per_template"):
                f.write("**Per-Template Analysis**\n\n")
                for t, v in rep["per_template"].items():
                    f.write(f"- **{t}**: Uses={v['uses']}, Next-Turn Improve Rate={v['next_turn_improve_rate']:.3f}\n")
            f.write("\n---\n\n")

    print(json.dumps(reports, indent=2))
    print(f"\n✅ Wrote Markdown report to {md_path}")

if __name__ == "__main__":
    # Pass command-line arguments (file paths) to main, if any
    main(sys.argv[1:])
