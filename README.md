# BPC-UIM Projekt

## Řešitelské týmy a výběr tématu

Řešiteli projektu budou týmy složené z maximálně 3 studentů/studentek.

Každý tým si zvolí jedno téma projektu dle libosti. Doporučujeme důkladně se seznámit s materiály k jednotlivým projektům (v e-Learningu ➔ Materiály k projektu).

Z každého týmu provede registraci k vybranému tématu v e-learningu právě jeden člen. Kapacita jednoho tématu je omezena na 11 týmů; pokud by kapacita nestačila, bude situace řešena individuálně. Student, který registraci provede, zároveň zašle na adresu Marina.Filipenska@vut.cz seznam členů svého týmu s předmětem:

* **BPC-UIM, projekt, sk. {jméno1, jméno2, jméno3}**

## Konzultace

Projekty bude možné konzultovat s přiřazeným vyučujícím. Konzultace si domlouvejte individuálně přímo s vyučujícím.

## Důležité termíny

Zaslání složení týmů a vybraného tématu projektu na Marina.Filipenska@vut.cz (informace za celý tým zasílá kontaktní osoba): ***do 12. 10. 2026***

Nejzazší termín odevzdání řešení (na 195715@vut.cz): ***6. 12. 2026, 23:59***

## Struktura repozitáře

Tento repozitář je šablona projektu. Soubor `train.csv` svého tématu vložte do složky `data/`.

```
├── testing.py         # vyhodnocení natrénovaného modelu (viz Odevzdání a nezávislé testování)
├── train.py           # trénování modelu
├── requirements.txt   # použité knihovny včetně verzí
├── config/            # nastavení experimentů
├── data/              # train.csv
├── dataio/            # načítání a ukládání dat
├── logs/              # záznamy z trénování
├── models/            # uložený natrénovaný model a naučené předzpracování
├── src/               # předzpracování, model, trénování
├── tests/             # jednotkové testy vašich funkcí
└── utils/             # pomocné funkce; metrics.py obsahuje výpočet matice záměn a MCC
```

Složka `tests/` je určena pro jednotkové testy vašeho kódu, kdežto `testing.py` vyhodnocuje hotový model na datech. Funkce `matice_zamen()` a `mcc()` můžete použít i při trénování a validaci (`from utils import matice_zamen, mcc`).

## Data

Všechna tři témata jsou úlohy binární klasifikace. Ke každému tématu dostanete soubor `train.csv` s trénovacími daty: každý řádek je jeden vzorek, sloupce jsou příznaky a poslední sloupec `target` je cílová proměnná (0/1). Význam sloupců popisuje datový slovník u každého tématu.

- Formát CSV: oddělovač `,`, kódování UTF-8, desetinná tečka.
- Pořadí sloupců i řádků je náhodné a nenese žádnou informaci.
- **Data obsahují chybějící hodnoty** (prázdné pole v CSV). Netýkají se binárních příznaků ani sloupce `target`.
- Hodnoty příznaků nebyly očištěny od odlehlých hodnot. Extrémní hodnoty jsou reálnou součástí dat a jejich ošetření je součástí úlohy.
- Pro řešení používejte pouze dodaná data.

## Odevzdání a nezávislé testování

Odevzdané algoritmy ***budou testovány na nezávislém testovacím souboru dat***, který mají k dispozici pouze vyučující. Testovací soubor má formát shodný s `train.csv`: stejné sloupce ve stejném pořadí, včetně sloupce `target`. Obsahuje chybějící hodnoty ve stejné míře a stejného charakteru jako trénovací data. Vaše řešení proto musí umět chybějící hodnoty ošetřit nejen při trénování, ale i při predikci.

Odevzdejte `.zip` archiv se všemi soubory, které jsou nutné pro spuštění vašeho projektu:

- `requirements.txt` s verzemi použitých knihoven,
- uložený natrénovaný model včetně naučeného předzpracování (např. imputace, škálování),
- `testing.py` vytvořený doplněním přiložené šablony,
- složku `utils/` a veškerý kód, který `testing.py` importuje,
- další potřebné soubory (např. kód, kterým jste model natrénovali).

Šablona `testing.py` načte testovací soubor, předzpracuje jej, zavolá váš model a vypíše matici záměn a MCC. Doplňte v ní pouze funkce `nacti_model()`, `predzpracuj()` a `predikuj()`:

- předzpracování testovacích dat (imputace, ošetření odlehlých hodnot, škálování apod.) používá parametry naučené na trénovacích datech a nesmí odstranit žádný řádek: predikce musí vzniknout pro každý vzorek,
- model se v `testing.py` znovu netrénuje, pouze se načte ze souboru a provede predikci (u neuronových sítí v režimu vyhodnocení, např. `model.eval()` v PyTorch),
- sloupec `target` slouží jen k výpočtu MCC a do predikce vstupovat nesmí,
- cestu k testovacímu souboru (`DATA_PATH`) vyučující při testování přepíše; skript proto nesmí záviset na jiných cestách mimo odevzdaný archiv.

Před odevzdáním si ověřte, že `testing.py` proběhne v čistém prostředí na souboru `train.csv`.

Výstupy odevzdá vybraný člen týmu (kontaktní osoba) za celý tým. O výsledcích testování se dozvíte v přehledové tabulce.

## Hodnocení

Úspěšnost klasifikace se u všech témat hodnotí pomocí matice záměn (confusion matrix) a z ní vypočítaného Matthewsova korelačního koeficientu (MCC):

$$
\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

MCC nabývá hodnot od −1 do 1: hodnota 1 znamená bezchybnou predikci, 0 predikci na úrovni náhody. Na rozdíl od accuracy je vypovídající i u nevyvážených tříd: model, který vždy predikuje většinovou třídu, může mít vysokou accuracy, ale jeho MCC je 0.

Řešení projektu bude hodnoceno soutěžní formou. Body se rozdělí do dvou kategorií:

* **Formální stránka (15 bodů):** Hodnotí se kvalita odevzdaného kódu, správná aplikace metod a celková funkčnost řešení.
* **Úspěšnost (10 bodů):** Všechna odevzdaná funkční řešení budou otestována na nezávislé sadě dat. Na základě dosaženého MCC bude sestaven žebříček, který určí počet získaných bodů (nejlepší tým 10 bodů, nejslabší 1 bod, ostatní budou ohodnoceni poměrově mezi těmito hodnotami). Každý tým soutěží pouze v rámci svého zadání.

***Hodnocení je nastaveno tak, aby podpořilo férovou soutěž mezi týmy a zároveň motivovalo k co nejlepším výsledkům.***

***Plagiátorství bude mít za následek neudělení zápočtu!***