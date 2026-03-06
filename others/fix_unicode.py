"""Fix double-encoded UTF-8 mojibake in rf_classifier_copy.py at byte level - pass 2"""

with open('rf_classifier_copy.py', 'rb') as f:
    raw = f.read()

# Remaining broken sequences found in pass 1
replacements = {
    # 🏆 trophy (broken) -> [BEST]
    b'\xc3\xb0\xc5\xb8\xc2\x8f\xe2\x80\xa0': b'[BEST]',
    # 🚀 rocket (broken variant) -> [START]
    b'\xc3\xb0\xc5\xb8\xc5\xa1\xe2\x82\xac': b'[START]',
    # 🔯 -> [TARGET]  or 🎯 variant
    b'\xc3\xb0\xc5\xb8\xc5\xbd\xc2\xaf': b'[TARGET]',
    # 🎲 dice variant
    b'\xc3\xb0\xc5\xb8\xc5\xbd\xc2\xb2': b'[RANDOM]',
    # 📈 chart (variant)
    b'\xc3\xb0\xc5\xb8\xe2\x80\x9c\xcb\x86': b'[CHART]',
    # 🔍 search right variant
    b'\xc3\xb0\xc5\xb8\xe2\x80\x9d\xc2\x8d': b'[SEARCH]',
    # 🔢 input numbers variant
    b'\xc3\xb0\xc5\xb8\xe2\x80\x9d\xc2\xa2': b'[NUM]',
    # 🔄 cycle/refresh variant
    b'\xc3\xb0\xc5\xb8\xe2\x80\x9d\xe2\x80\x9e': b'[CV]',
    # \xc2\x90 -> empty (control char)
    b'\xc2\x90': b'',
}

for old, new in replacements.items():
    raw = raw.replace(old, new)

with open('rf_classifier_copy.py', 'wb') as f:
    f.write(raw)

print('Pass 2 done. Remaining broken sequences fixed.')
