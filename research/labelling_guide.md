# Labelling guide: Notion App Store reviews

**Status: draft for approval.** This exact text is also the instruction given to the LLM annotator,
so the human and the LLM label against the same definitions.

## Purpose

We are building a test set for choosing the sentiment model. Every review gets **two labels**:

1. **Sentiment**: the author's overall stance towards the app: `positive`, `neutral` or `negative`.
2. **Complaint**: does the review mention a problem, something missing, or something that should
   change? `yes` or `no`.

They are separate questions. A happy user can still complain ("Love it, wish it had dark mode"), and
that complaint matters for the "areas of improvement" analysis.

## What you see

- The review **title and text**. For non-English reviews, an English translation is shown below the
  original.
- **Star ratings are hidden on purpose.** We want labels from the text alone, because stars and text
  disagree in about 20% of app reviews.

## Label 1: Sentiment

Use one test: **"Would this person recommend the app as it is right now?"**

| Label | When | Typical signals |
|---|---|---|
| **positive** | Yes, they would. The overall experience is good, even with minor wishes. | praise, gratitude, "life changer", recommendations |
| **negative** | No, they would not. Problems dominate or make the app bad or unusable for them. | frustration, "ruined", "unusable", "uninstalling", bugs that block use, sarcasm |
| **neutral** | You can't tell. There is no clear stance. | questions, plain factual statements, confusion without frustration, truly balanced mixed reviews, gibberish |

Rules:
- **Judge the whole review, not one word.** "Love Notion, but the app is so buggy. It's impossible to
  comment on anything" is **negative**: the love is for the past, the experience now is bad.
- **Read sarcasm for what it means.** "thanks for logging me out after updating. appreciate it" is
  **negative**.
- **Nostalgia is not praise.** "This was one of the best note taking apps. But with recent updates you
  ruined it" is **negative**.
- **A minor wish does not flip a positive review.** "Love it, but I hate that there are 'limited
  blocks'" is **positive**, with complaint = yes.
- **Neutral is a real answer, not a fallback for "hard".** If the stance is clear but mixed, decide which
  side wins using the recommendation test.
- **Ignore spelling, grammar, language and length.** "review. good" is **positive**.

## Label 2: Complaint (yes / no)

**Yes** if the review mentions *anything* that doesn't work, is missing, is too expensive, is confusing,
or should change. This includes:
- bugs, crashes, lag, sync problems;
- missing features and requests ("please add", "wish it had", "should support");
- pricing and paywall complaints;
- complaints about AI features, updates, design changes, support;
- problems stated without emotion ("Latex equations are not viewable on iPad").

**No** if the review is only praise, only a question, or only a neutral description.

## Hard cases (labels decided)

These are real reviews from the pool. None of them are in the test set.

| Review | Sentiment | Complaint | Why |
|---|---|---|---|
| "Love it, but. I hate that there are 'limited blocks'" | positive | yes | Overall a fan; the limit is a wish |
| "Hate the AI. Love the app, used it for ages but I hate the use of AI now" | negative | yes | The title and the main point are the complaint; the love is for the past |
| "Ruined a great app. Notion was untouchable, then they ruined it by becoming a slop-first app" | negative | yes | Nostalgia, not praise |
| "Cmon guys. Horrible crashes. Slow. Awful phone app. But fantastic format. If your tech support was any good it would be 5 stars" | negative | yes | The problems dominate the current experience |
| "Great app. Can be complicated at times" | positive | yes | Recommends it; complexity is a mild complaint |
| "Umm? it comes highly recommended. i'm not sure what it is exactly. i wish i knew how to use it" | neutral | yes | No stance yet; difficulty of use counts as a complaint |
| "Latex equations are not viewable on iPad." | neutral | yes | A factual problem report with no overall stance |
| "I keep getting error 400 do you know what that means?" | neutral | yes | A question about a problem |
| "is ai really necessary? title^" | negative | yes | A rhetorical question expressing dislike |
| "thanks for logging me out after updating. appreciate it" | negative | yes | Sarcasm |
| "Mrs. Helena" | neutral | no | No content |
| "Amazing!! Amazing!!" | positive | no | |

## When unsure

Pick the best label anyway and mark the review **unsure**. Unsure reviews are analysed separately: they
show where the task itself is ambiguous, which also caps how well any model can do.

## In the labelling tool

| Key | Action |
|---|---|
| `1` | positive |
| `2` | neutral |
| `3` | negative |
| `c` | toggle complaint |
| `u` | toggle unsure |
| `Enter` | save and go to the next review |
| `←` | go back |

Progress is saved after every review.
