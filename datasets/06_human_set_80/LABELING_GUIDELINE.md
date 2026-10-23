# Human labeling guideline — messy-query external-validity set (EN)

## Goal
For each question, decide what a **correct financial assistant should do** — judged only from
the question as the user wrote it. This human, system-agnostic label is the ground truth we
score all systems (VerifiQuant, R-PAL, CoT) against; it removes the "traps came from our own
contracts" objection.

## The four labels — mark just the letter: **A / R / C / O**
- **A — ANSWER**: complete, unambiguous, a standard finance computation whose right response is
  one number. *(well-posed, computable)*
- **R — REQUEST-INFO**: a required numeric **value** is missing from the text; the assistant
  should ask for that specific value. *(well-posed, but data missing)*
- **C — CLARIFY**: all values are present, but a genuine **interpretation** choice would change
  the number; the assistant should ask which interpretation (or state its assumption).
  *(computable once you pick an interpretation)*
- **O — OUT-OF-SCOPE**: the question is **not a determinate single-number computation at all** —
  advice/opinion, a forecast, or it needs data/a model that can't be obtained here.
  *(the ask itself is not "compute one number")*

## Decision flow — go top to bottom, stop at the FIRST "yes"
1. Is the **ask itself** not "compute one number"? — advice/opinion ("should I…", "is it a good
   deal"), a forecast ("will it go up"), or it needs a method / data / model you cannot get from
   the text → **O**
2. Is a required numeric **value missing** from the text (a number you'd need but that isn't
   given)? → **R**
3. Are all values present but an **interpretation** choice would change the number? — percent vs
   decimal · begin vs end / which period · which metric · 360 vs 365 · gross vs net · per-share
   vs total → **C**
4. Otherwise — complete, unambiguous, one determinate number → **A**

### The two contrasts people trip on
- **R vs C** — R = a *number* is missing (ask for the value). C = the numbers are all there, but
  *which interpretation* is unclear (ask which one).
- **C vs O** — one-question test: *"If I could ask exactly one clarifying question about
  interpretation, could I then compute one exact number from the data given?"*
  **Yes → C.  No → O.** (O is advice, a forecast, or needs data/a model — no clarifying question
  about interpretation turns it into a computed number.)

## Golden rules
1. Judge from the **question + context only**, as if you were the assistant. Ignore any system's
   output or design.
2. **Don't infer an unstated number from nearby figures.** If a partial table/excerpt supplies
   several inputs but omits one value required by the formula, the data is still missing (**R**).
   Name the specific value the assistant should request.
3. **User confusion ≠ task ambiguity.** "I'm not sure how to compute it" but all data present and
   standard → **A**. Their uncertainty does not change the label.
4. **The label is what the assistant should *do*, not whether the question's own numbers/claims
   are correct.** A "does this check out?" verification with full data → **A** (compute the true
   value and flag if the claim is wrong). *e.g.* "A headline says the 95% VaR is ~$3.3; does that
   check out? [full inputs given]" → **A** (real VaR ≈ $1.65).
5. **Scope is about the question, not your skill.** Don't mark **O** just because the formula is
   advanced or you'd have to look it up. Hard-but-computable with all inputs given — duration VaR,
   bond convexity, option Greeks — is **A**. Use **O** only when a competent finance person would
   say the *question itself* is not a determinate computation (advice, ill-posed, or needs
   unavailable data/models).
6. One label per item. Give a **one-line reason**. If genuinely torn, add **?** (goes to a
   disagreement pool rather than being forced).

## Examples
- **A** — "Revenue 500 (2022) → 550 (2023): YoY % change?" · "Face 1,000, 5% annual coupon, 3y,
  yield 4%: price?" · "Headline says VaR ≈ $3.3, does it check out? [inputs given]".
- **R** — "Revenue is 520 and operating cost is 390; what's ROA?" (average assets missing) ·
  "Face value 1,000, price 970, calculate money-market yield" (days to maturity missing).
- **C** — "Annualize this 90-day return" (360 vs 365 basis) · "What's the growth?" with several
  years given (YoY of which year? vs CAGR) · "Give me the debt ratio" (debt/equity vs debt/assets).
- **O** — "Should I invest in this project?" (advice) · "Will the stock rise next quarter?"
  (forecast) · "Price this with a Heston model" (needs simulation; no closed form from the given
  inputs) · "What's my VaR?" with no method and no inputs (unanswerable as posed).

> Contrast on the *same* project: "What's the NPV of these cash flows?" → **A**; "Which return
> metric do you want here?" (IRR vs simple, both computable) → **C**; "Is this project worth it?"
> → **O** (a worth/decision judgment, not a number).

## How to fill
- **Make your own tab (分頁) and work there.** In the shared sheet, duplicate the question tab,
  rename it with your name, and label in *your* copy — don't edit anyone else's tab. (In the
  Markdown/CSV file, just fill the `label` and `reason` columns.)
- Put **A / R / C / O** (add **?** if torn) + a one-line reason. For **A** items, jotting the
  intended metric (e.g. "YoY % change") helps our later scoring.
- Label **independently** — don't compare with other annotators until each of you has finished.
