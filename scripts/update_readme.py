import csv, os, re, datetime
HEVAL='results/heval_metrics.csv'; GSM='results/gsm8k_metrics.csv'; README='README.md'
def read_metric(p):
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))
def table(rows, cols):
    out = ["|"+"|".join(cols)+"|", "|"+"|".join(["---"]*len(cols))+"|"]
    for r in rows: out.append("|"+"|".join(r.get(c,"") for c in cols)+"|")
    return "\n".join(out)
def main():
    he = read_metric(HEVAL); gs = read_metric(GSM)
    ts = datetime.datetime.utcnow().isoformat()+"Z"
    he_rows = [r for r in he][-5:]
    gs_rows = [r for r in gs][-5:]
    block = f"""<!-- RESULTS_START -->
**Latest Results (UTC {ts})**

**HumanEval (exec-based)**  
{table(he_rows, ["dataset","split","model","temperature","k","metric_name","metric_value","timestamp"])}

**GSM8K (full-output parsed)**  
{table(gs_rows, ["dataset","split","model","temperature","metric_name","metric_value","timestamp"])}
<!-- RESULTS_END -->"""
    txt = open(README,'r',encoding='utf-8').read()
    if "<!-- RESULTS_START -->" in txt:
        txt = re.sub(r"<!-- RESULTS_START -->[\s\S]*?<!-- RESULTS_END -->", block, txt, flags=re.M)
    else:
        txt += "\n\n"+block+"\n"
    open(README,'w',encoding='utf-8').write(txt)
if __name__=="__main__": main()
