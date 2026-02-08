# Symulacja rozgałęzień wyładowania elektrycznego (piorun)

## Opis

Projekt składa się z dwóch programów napisanych w języku Python.

**`lightning_sim.py`** realizuje dwuwymiarową symulację wzrostu kanału wyładowania
elektrycznego inspirowanego zjawiskiem pioruna. Wzrost kanału sterowany jest
rozkładem pola elektrycznego, który w każdym kroku wyznaczany jest z równania
Laplace’a. Symulacja pozwala badać wpływ własności ośrodka na kształt
rozgałęzień poprzez zastosowanie różnych scenariuszy.

**`make_mp4.py`** służy do złożenia zapisanych klatek symulacji (plików PNG)
w animację w formacie MP4 przy użyciu programu `ffmpeg`.

---

## Pliki

- **`lightning_sim.py`**  
  Główny program symulacyjny.  
  Odpowiada za:
  - obliczanie potencjału elektrycznego (równanie Laplace’a),
  - wyznaczanie pola elektrycznego,
  - stochastyczny wzrost kanału wyładowania,
  - zapis klatek symulacji do katalogu `output/`.

- **`make_mp4.py`**  
  Program pomocniczy do tworzenia animacji.  
  Odpowiada za:
  - zebranie plików PNG z katalogu `output/`,
  - posortowanie ich według numeru kroku,
  - złożenie ich w plik MP4 przy pomocy `ffmpeg`.

- **`output/`**  
  Katalog tworzony automatycznie przez program symulacyjny do przechowywania:
  - klatek symulacji w formacie PNG,
  - gotowych plików wideo MP4.

---

## Sposób użycia

### 1. Uruchomienie symulacji

Podstawowe uruchomienie symulacji nie wymaga podawania żadnych parametrów poza wyborem scenariusza.
Zalecane jest rozpoczęcie od ustawień domyślnych.

**Podstawowe polecenie**

```bash
python lightning_sim.py
```

---

### 2. Dostępne parametry symulacji

| Parametr        | Opis                                     | Wartość domyślna       | Zalecane do testów        |
| --------------- | ---------------------------------------- | ---------------------- | ------------------------- |
| `--scenario`    | Wybór scenariusza środowiska             | `1`                    | `1`, `2` lub `3`          |
| `--seed`        | Ziarno losowości (powtarzalność wyników) | losowe                 | liczba naturalna `< 10²⁰` |
| `--H`           | Wysokość siatki                          | `160`                  | `120–200`                 |
| `--W`           | Szerokość siatki                         | `240`                  | `180–300`                 |
| `--max_steps`   | Maks. liczba kroków wzrostu              | `2500`                 | `1500–3000`               |
| `--eta`         | Czułość wzrostu na pole elektryczne      | zależna od scenariusza | `1.2–2.0`                 |
| `--relax_iters` | Iteracje rozwiązania Laplace’a na krok   | `60`                   | `40–100`                  |
| `--omega`       | Współczynnik nadrelaksacji SOR           | `1.85`                 | `1.6–1.9`                 |
| `--save_every`  | Co ile kroków zapisywać klatkę           | `15`                   | `5–30`                    |
| `--no_images`   | Brak zapisu klatek PNG                   | wyłączone              | —                         |


---

### 3. Przykładowe konfiguracje

| Cel                                    | Polecenie                                        |
| -------------------------------------- | ------------------------------------------------ |
| Najprostsza symulacja                  | `python lightning_sim.py --scenario 1`           |
| Bariera izolacyjna                     | `python lightning_sim.py --scenario 2`           |
| Kanał preferowany                      | `python lightning_sim.py --scenario 3`           |
| Ten sam wynik przy każdym uruchomieniu | `python lightning_sim.py --scenario 1 --seed 42` |
| Więcej rozgałęzień (większa losowość)  | `python lightning_sim.py --scenario 1 --eta 1.2` |
| Bardziej „ostry” kanał                 | `python lightning_sim.py --scenario 1 --eta 2.0` |
| Mniejszy rozmiar (szybsze liczenie)    | `python lightning_sim.py --H 120 --W 180`        |

---

### 4. Wyniki symulacji

Po zakończeniu działania programu w katalogu `output/` znajdują się:

| Plik            | Znaczenie                                     |
| --------------- | --------------------------------------------- |
| `scX_00000.png` | Początkowy stan symulacji                     |
| `scX_YYYYY.png` | Kolejne etapy wzrostu                         |
| `scX_FINAL.png` | Stan końcowy (po dotarciu do dolnej krawędzi) |

gdzie **X** oznacza numer scenariusza, a **Y** kolejne kroki symulacji.

---

### 5. Tworzenie animacji MP4

Po wykonaniu symulacji można utworzyć film z zapisanych klatek.

**Domyślne polecenie**

```bash
python make_mp4.py
```

#### Parametry animacji

| Parametr | Opis                     | Wartość domyślna | Zalecane |
| -------- | ------------------------ | ---------------- | -------- |
| `--fps`  | Liczba klatek na sekundę | `30`             | `10–30`  |

**Przykład**

```bash
python make_mp4.py --fps 10
```

W katalogu `output/` zostanie utworzony plik:

**`sc1.mp4`, `sc2.mp4` lub `sc3.mp4`**



