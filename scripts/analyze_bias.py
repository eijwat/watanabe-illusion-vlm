import json, re, sys

N_DOTS = 11
VERIDICAL_TOP = 1      # Q1 from-top veridical
VERIDICAL_BOT = 11     # Q2 from-bottom veridical
HUMAN_Q1, HUMAN_Q2 = 4.63, 6.33   # paper_v12 human means (N=133)
ORD = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
       "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11}

def parse(text):
    q = re.findall(r'"the\s+([\w-]+)\s+dot\s+from\s+the\s+(top|bottom)"', text, re.I)
    src = q[-1] if q else None
    if src is None:
        m = re.findall(r'([\w-]+)\s+dot\s+from\s+the\s+(top|bottom)', text, re.I)
        src = m[-1] if m else None
    if src is None: return None
    w, frame = src[0].lower(), src[1].lower()
    if w == "last": k = N_DOTS
    elif w in ("second-to-last","secondtolast"): k = N_DOTS - 1
    else:
        mn = re.match(r'(\d+)', w)
        k = int(mn.group(1)) if mn else ORD.get(w)
    if k is None: return None
    return k, frame

def to_top(k, frame):   # express as 'from top' index
    return k if frame == "top" else (N_DOTS + 1 - k)
def to_bot(k, frame):
    return k if frame == "bottom" else (N_DOTS + 1 - k)

data = json.load(open(sys.argv[1]))
print(f"model: {data['model_id']}   N={len(data['trials'])} (pilot)\n")
q1_top, q2_bot, sums = [], [], []
for t in data["trials"]:
    if "Q1" not in t: continue
    p1, p2 = parse(t["Q1"]["response"]), parse(t["Q2"]["response"])
    a_top = to_top(*p1) if p1 else None
    b_bot = to_bot(*p2) if p2 else None
    if a_top is not None: q1_top.append(a_top)
    if b_bot is not None: q2_bot.append(b_bot)
    s = (a_top + (N_DOTS+1-b_bot)) if (a_top and b_bot) else None  # both as top
    if a_top and b_bot: sums.append(a_top + b_bot)
    print(f"trial {t['trial_idx']}: "
          f"Q1 raw={p1} -> {a_top} from-top (veridical {VERIDICAL_TOP}; "
          f"{'+'+str(a_top-VERIDICAL_TOP) if a_top else '?'} = illusion dir)" )
    print(f"          Q2 raw={p2} -> {b_bot} from-bottom (veridical {VERIDICAL_BOT}; "
          f"{b_bot-VERIDICAL_BOT if b_bot else '?'} = illusion dir)  | Q1+Q2(top+bot)={a_top+b_bot if (a_top and b_bot) else '?'}")

def mean(x): return sum(x)/len(x) if x else float('nan')
print("\n--- summary ---")
print(f"Q1 from-top:    values={q1_top}  mean={mean(q1_top):.2f}  (veridical=1, human={HUMAN_Q1})")
print(f"Q2 from-bottom: values={q2_bot}  mean={mean(q2_bot):.2f}  (veridical=11, human={HUMAN_Q2})")
print(f"Q1 all > veridical(1)?  {all(v>VERIDICAL_TOP for v in q1_top)}  -> pulled toward canonical/middle")
print(f"Q2 all < veridical(11)? {all(v<VERIDICAL_BOT for v in q2_bot)}  -> pulled toward canonical/middle")
print(f"Q1+Q2 consistency (ideal=12): {[s for s in sums]}  mean={mean(sums):.2f}")
