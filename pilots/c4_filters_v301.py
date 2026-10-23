"""Port of TFDS C4 v3.0.1 (commit f411fae) clean_page + badwords filter, without TensorFlow.
Faithful to research-empirical-methods/reference-code/tfds_c4/c4_utils_v3.0.1_f411fae.py except:
 - sentence tokenizer: NLTK PunktTokenizer('english') from punkt_tab (NLTK>=3.9 no longer loads the punkt pickle TFDS used).
 - dedup (3-sentence spans across the corpus) and langdetect are NOT applied here."""
import re
from nltk.tokenize.punkt import PunktTokenizer
_TOK=PunktTokenizer("english")
MIN_WORDS_PER_LINE=5; MIN_NUM_SENTENCES=3; MAX_WORD_LENGTH=1000
END_MARKS=(".","?","!",'"'); ELLIPSIS="..."
POLICY=["terms of use","privacy policy","cookie policy","uses cookies","use of cookies","use cookies"]
CITATION=re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")
def clean_page(text):
    """Returns (cleaned_text or None, reason)."""
    valid=[]; n_sent=0
    for line in text.splitlines():
        line=line.strip()
        if any(len(w)>MAX_WORD_LENGTH for w in line.split()): continue
        line=CITATION.sub("",line)
        if not line.endswith(END_MARKS) or line.endswith(ELLIPSIS): continue
        if len(line.split())<MIN_WORDS_PER_LINE: continue
        low=line.lower()
        if "lorem ipsum" in low: return None,"loremipsum"
        if "javascript" in low: continue
        if "{" in line: return None,"curly"
        if any(p in low for p in POLICY): continue
        n_sent+=len(_TOK.tokenize(line)); valid.append(line)
    if n_sent<MIN_NUM_SENTENCES: return None,"too_few_sentences"
    return "\n".join(valid).strip(),"passed"
def load_badwords(path):
    words=[w.strip() for w in open(path,encoding="utf-8") if w.strip()]
    return re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))  # v3.0.1: not escaped
