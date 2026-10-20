# Review notes: rows where the proposed manual label differs from the automatic label

Labels in review_sample.csv were pre-filled by Claude Opus 5.5 and must be checked by the author.

| row | id | question | answer | auto | proposed | why |
|---|---|---|---|---|---|---|
| 4 | 3061785 | What is Alex Reid's occupation? | mixed martial artist | incorrect | ambiguous | Alex Reid (English actress) vs Alex Reid (British MMA fighter): answer is true for a same-name entity |
| 12 | 5905586 | What genre is Tharg the Mighty? | comics | incorrect | ambiguous | Tharg the Mighty is a character in a sci-fi comic; 'comics' names the medium, not wrong, not the genre |
| 13 | 228067 | What genre is Avatar: The Last Airbender? | fantasy | incorrect | correct | Avatar: The Last Airbender is a fantasy animated series; 'fantasy' is a correct genre the alias list lacks |
| 15 | 988268 | What genre is Uno? | card game | incorrect | ambiguous | 'Uno' the card game vs the Muse song (gold); answer true for another entity |
| 16 | 2946598 | What genre is Will? | drama | incorrect | ambiguous | 'Will' is ambiguous (gold: an electronic album); 'drama' fits a film of that name |
| 17 | 5815875 | What genre is The Fallen? | young adult | incorrect | ambiguous | 'The Fallen' is a YA book series (Sniegoski) and a video game (gold); answer true for another entity |
| 24 | 4645372 | In what country is Japan–South Korea relations? | South Korea | incorrect | ambiguous | Ill-posed question: Japan-South Korea relations concern both countries |
| 54 | 6451484 | Who was the screenwriter for Scooby-Doo? | Joe Ruby | incorrect | ambiguous | Joe Ruby co-wrote the original Scooby-Doo series; gold refers to the 2002 film (James Gunn) |
| 58 | 2454962 | Who was the screenwriter for The Holes? | Louis Sachar | incorrect | ambiguous | Louis Sachar wrote 'Holes' (2003 film); gold is the French film 'The Holes' (1974) |
| 62 | 5913921 | Who was the screenwriter for The Boys? | Eric Kripke | incorrect | ambiguous | Eric Kripke developed the TV series 'The Boys'; gold is the 1962 Finnish film |
| 76 | 470734 | Who was the composer of Let It Be? | Paul McCartney | incorrect | correct | Paul McCartney wrote 'Let It Be' (credited Lennon-McCartney); gold lists only Lennon |
| 77 | 790271 | What is the religion of Muscat and Oman? | Islam | incorrect | correct | Muscat and Oman: Ibadi Islam; 'Islam' is correct at coarser granularity |
| 85 | 672890 | Who is the author of Capital? | Karl Marx | incorrect | ambiguous | Karl Marx wrote 'Capital' (Das Kapital); gold is John Lanchester's novel 'Capital' |
