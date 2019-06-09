from bertsum import BSummarizer

b = BSummarizer('/home/beato/bert-server/models/bertsum_state_dict.pt')

# https://www.rappler.com/nation/232644-pnp-arrest-davao-today-columnist-margarita-valle-mistaken-identity
b.summarize("""The arrest of Davao Today columnist Margarita Valle on Sunday, June 9, was a case of mistaken identity, according to the Philippine National Police (PNP).
PNP spokesman Colonel Bernard Banac said 61-year-old Valle is set for release after a witness confirmed she only has a "major resemblance" to the actual suspect.
"Upon arrival of the witness who physically identified the subject, witness further averred that the suspect has major resemblance but is not the actual suspect who is the subject of the warrant," Banac said.
Valle was arrested by cops and soldiers at the Laguindingan Airport in Misamis Oriental on Sunday based on arrest warrants for charges of "multiple murder with quadruple frustrated murder," destruction of government property, and arson.
The Davao City chapter of the National Union of Journalists of the Philippines (NUJP) condemned the arrest, saying it shows the "possibility that journalists working in communities may be a future target of the threats, harassment, and killings as Mindanao remains under the power of martial law."
Valle, according to the NUJP, is a seasoned journalist with "vast experience in reporting various issues in Mindanao," including advocating for human rights and peace development.
The group blamed the Duterte administration for Valle's arrest and other recent attacks on journalists.
"[The government] foolishly believes in the conspiracy theory of people linking with the Communist Party, using [the] same pretext and plot weaved by past administrations," said the NUJP.
The College Editors Guild of the Philippines also condemned Valle's arrest, saying that it shows the Duterte administration's efforts to silence "people who aspire for the common good.""")

# https://www.theguardian.com/film/2017/feb/19/john-wick-chapter-2-review-keanu-reeves-full-force
b.summarize("""John Wick is a man of focus, commitment and sheer will. The stories you hear about this man, if nothing else, have been watered down”, or so the legend goes. In the follow-up to the slick 2014 action-thriller, former hitman Wick (Keanu Reeves, more magnetic than ever) is ushered out of retirement once again. Attempting to find peace as part of his new life in upstate New York, he is forced to honour the blood oath he once made to Italian playboy Santino D’Antonio (Riccardo Scamarcio).
Wick’s bounty lives in Rome, the perfect setting for a Bond-style montage of Reeves trying on tailored suits and meeting a “sommelier” who deals firearms instead of fine wines, and a breathlessly violent chase through the catacombs, complete with thrashing heavy metal soundtrack. With their jewel-toned neon lighting and often elegant settings (look out for an art gallery cameo and a gorgeous ancient Roman bath), there’s poetry and pathos in the film’s balletic fight sequences, even if the body count begins to become difficult to stomach as the film races towards its bloody climax.
The Wick franchise aspires to Hong Kong-style martial arts films, differentiating it from Bond or Bourne. An adrenaline-pumping blockbuster polished to near perfection, save its sequel-baiting conclusion.""")
