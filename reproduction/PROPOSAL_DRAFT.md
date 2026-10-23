# Documenting large webtext corpora, a case study on C4

*Refined draft. Your wording and tone are kept. Edits fix typos and grammar, use the paper's own terms where
the original was unclear, and tie each claim to the section of the paper it comes from. Every change is
listed at the end, and the original text is preserved verbatim below that.*

---

## Which problem you chose, and why

The problem that interested me the most right from the get-go was option C. I am someone who has fine-tuned a few
models and studied the post-training effects that arise as a result of biases in the data, and C4 was something I had
heard about, so I started reading this paper out of curiosity and liked it. I think this is particularly important
because more and more of the models that we interact with are built on the same foundational data, which we assume is
an accurate representation of the internet. If that representation itself is wrong, then it is worth spending time to
analyse why, and maybe fix it, instead of spending a lot of time afterwards on debiasing and correcting model alignment.

## The specific claim or question you are going to test

Out of the several claims I would like to test, here are the ones that I think will be doable given the time, compute
and resources:

1. **Ethnicity bias in the data carries into models (§4.3).** The paper claims that the data contains bias that differs
   by ethnicity, i.e. some ethnicities are simply associated with more negative connotations, and that this in turn
   translates into the biases of models trained on it. (In the paper: UnifiedQA, a T5 model pre-trained on C4, favours
   "Jewish" over "Arab", and C4 paragraphs mentioning "Jewish" carry more positive sentiment words than those
   mentioning "Arab".)
2. **C4's filtering excludes some demographics unfairly (§5.2–5.3).** The paper also claims that the data removed by
   C4's filtering (the "bad words" blocklist) tends to exclude certain demographics more than others, for example
   documents mentioning sexual orientations, and documents written in African American and Hispanic-aligned English.
3. **Whose English is in C4 (§3.3, §5.3).** The paper also argues that, despite there being many English-speaking
   countries, most of the data is geared towards a narrow slice of English: 97.8% of documents are classified as
   White-aligned English (WAE), and 51.3% of pages are hosted in the US.

The questions I want to test are:

- Does the bias angle hold up? If it does, does running our data through a sentiment classifier (rather than counting
  words from a sentiment lexicon) help, or change the picture?
- Does the bias differ across different categories of documents (for example news, forums or patents, or across
  web domains)?
- Do LLM-generated or machine-translated documents amplify that bias?

Next, I am also going to see whether there are significant differences between the biases of documents that are fully
or partially artificially generated and those written by humans. I also want to see how the language distribution and
the most frequently used words in the training data have changed with the rise of content produced by LLMs, and
whether that affects the quality of the training data. This claim is a bit ambitious, and would only be possible if I
can get access to a more recent version of C4 (or the same C4 recipe applied to a newer crawl) in the same format.

## What you expect to find before you run anything

Before I run anything, one thing I would expect is that, in general, the bias in the training data has become more
covert, i.e. standard sentiment classifiers are not able to catch it, and that the distribution of the training data
has changed over time. Since we are dealing with a snapshot of the entire internet, there will be sites that are more
biased towards one race, country, etc., and I would expect that they make up a substantial part of what we consider
the bias in the dataset itself.

## What result would change your mind

One thing that would really change my mind is if the debiasing methods applied to LLMs have actually decreased the
bias, and if the bias against or for certain ethnicities that the paper reports was actually a matter of
over-sampling or under-sampling, so that standard rebalancing techniques like SMOTE (or, for text, re-weighting which
sites and domains are sampled) would fix it.

---

## What was changed, and why

| Original | Refined | Why |
|---|---|---|
| "finetuned", "post training effects that arise as a result of the biases of the data" | "fine-tuned", "post-training effects that arise as a result of biases in the data" | Spelling and grammar |
| "liked the same", "fix the same" | "liked it", "fix it" | "the same" as a pronoun reads as a typo in formal writing |
| "the data extracted as a part of the c4 filtering" | "the data removed by C4's filtering (the 'bad words' blocklist)" | The paper's claim (§5) is about what the filter **removes**, not what it extracts |
| "geared towards EAD english" | "White-aligned English (WAE)", with the paper's 97.8% and 51.3% figures | "EAD" is not a term in the paper. §5.3 uses WAE (from Blodgett et al. 2016); §3.3 gives the US-hosting share |
| One long paragraph of claims | Numbered claims 1–3, each tagged with its section | So each claim can be checked against the paper |
| "running our data through a sentiment classifier helps?" | "does running our data through a sentiment classifier (rather than counting words from a sentiment lexicon) help, or change the picture?" | The paper uses word lexicons, so the classifier is the contrast being tested |
| "different categories of document" | adds examples (news, forums, patents, web domains) | The paper already shows a domain effect (NYT vs Al Jazeera), so this anchors the question |
| "a more-recent version of c4 available somewhere" | adds "(or the same C4 recipe applied to a newer crawl)" | No dated C4 snapshots exist; the TFDS C4 builder can be run on newer Common Crawl dumps |
| "smote" | "SMOTE (or, for text, re-weighting which sites and domains are sampled)" | SMOTE synthesises feature vectors for classifier training. For a text corpus, the equivalent is resampling sources. Your idea is kept and made concrete |

---

## Original draft (verbatim)

> Documenting large webtext corpora, a case study on C4.
>
> - Which problem you chose, and why.
> The problem that interested me the most right from the get-go was option C. I am someone who has finetuned a few models and studied the post training effects that arise as a result of the biases of the data and C4 was something I had heard about so I started reading this paper out of curiosity and liked the same. I think this is particularly important because more and more of the models that we interact with are built on the same foundational data that we assume is the accurate representation of the internet, but if that representation itself is wrong then it is worth spending time to analyze why and maybe fix it instead of spending a lot of time after debiasing and correcting model alignment
>
> - The specific claim or question you are going to test.
> Out of the several claims I would like to test, here are some that I think will be doable given the time, compute and the resources. The paper claims that there exists some bias in the data which differs by ethnicity i.e. for some ethnicities, they are just naturally associated with more negative connotations and this in turn translates towards the biases in models. The paper also claims that the data extracted as a part of the c4 filtering also tends to exclude certain demographics more unfairly and the paper also argues that  despite there being a lot of english speaking countries, most of the data is geared towards EAD english. The claims I would like to test are whether or not the bias angle stands true, if it does then running our data through a sentiment classifier helps? Nextly, does the bias differ across different categories of document and whether or not having LLM generated documents or machine translated documents amplify that bias?
>
> Next, i am also going to see if there are any significant differences between the biases of documents which are either artificially generated fully or partially and the ones which are generated by humans. I also want to see how the language distribution and the frequently used words across the training data has changed given the rise of content produced by llms and whether that affects the quality of training data or not although this claim is a bit ambitious and would only be possible if i get access to a more-recent version of c4 available somewhere in the same format
>
> - What you expect to find before you run anything.
>
> Before I run anything, one thing I would expect is that in general, the bias in the training data has become more covert i.e. standard sentiment classifiers are not able to catch it and even the distribution of the training data has changed over time. Since we are dealing with the snapshot of the entire internet, there will exist sites which are more biased towards one race, country etc and I would expect that it makes up a substantial amount of what we consider the bias in the dataset itself
> - What result would change your mind.
>
> One thing that would really change my mind probably would be the fact that the debiasing methods applied on the llms have actually decreased the bias, and the bias against or for certain ethnicities that was accounted for in the paper was actually a matter of over sampling or under sampling and standard sampling techniques like smote would probably fix the same
