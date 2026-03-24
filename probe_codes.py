import pandas as pd, glob
paths = glob.glob('Data/**/*.csv', recursive=True)
seen = set()
for p in paths:
    try:
        df = pd.read_csv(p)
        key = (df['equipment_code'].iloc[0], df['exercise_code'].iloc[0], df['quality_code'].iloc[0])
        folder = p.replace('\\','/').split('Data/')[-1].rsplit('/',1)[0]
        if key not in seen:
            seen.add(key)
            print(f"eq={key[0]} ex={key[1]} q={key[2]}  |  {folder}")
    except: pass
