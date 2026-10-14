---
name: asd-ste100
description: Write and rewrite technical English in a simplified, controlled style (ASD-STE100-inspired) for agent-facing text, tool descriptions, error messages and documentation.
when_to_use: writing or rewriting technical prose, tool descriptions, error messages, PR descriptions, code comments, or agent-to-agent instructions where ambiguity has a real cost
tags: ste100, simplified-technical-english, controlled-language, documentation
---

# ASD-STE100-inspired Simplified Technical English

This runbook gives paraphrased writing rules inspired by ASD-STE100 Simplified Technical English. It is not the standard. For the full dictionary and official rule text, see the official site (https://www.asd-ste100.org/). Do not copy text from the standard into this file.

## Rules

1. One instruction per sentence. Keep instructions to 20 words or fewer.
2. Keep descriptions to 25 words or fewer per clause. Split longer clauses into two short sentences.
3. Use active voice and present tense for procedures and instructions.
4. Use one word for one meaning. Do not use synonyms for a defined term.
5. Define each new technical term once. Use that exact term everywhere after that.
6. Avoid hedging words such as "may", "might", "could" and "perhaps" unless a value is truly uncertain. State the uncertainty once, plainly.
7. Put the action first: "Run the tests." Not "The tests should probably be run."
8. Do not use idioms, slang or figures of speech.

## Technical names

Project names, tool names, API names and file names are Technical Names. Write them exactly as the code writes them. Do not rephrase them.

## Procedure

1. Read the source text and list each technical term it uses.
2. Rewrite each sentence to follow rules 1 to 8.
3. Check each sentence against the word limits.
4. Check that each term appears with one spelling only.
5. Return the rewritten text. Mention any term you could not resolve.

## Verification checklist

- Each instruction is one sentence of 20 words or fewer.
- Each clause is 25 words or fewer.
- Each instruction uses active voice where possible.
- Each defined term has one spelling.
- No hedging words remain unless a value is uncertain.

## References

- ASD-STE100 official site: https://www.asd-ste100.org/
- Issue 9 (January 2025) is the current edition.
