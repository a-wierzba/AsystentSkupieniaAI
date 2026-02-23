# 🧠 Asystent Skupienia AI

Projekt w ramach szkolenia dla nauczycielek i nauczycieli Lekcja: AI. Aplikacja wykorzystuje Computer Vision (OpenVINO) do analizy uwagi ucznia w czasie rzeczywistym.

## Funkcje
- Wykrywanie twarzy i emocji.
- Analiza kątów głowy (Yaw, Pitch) do oceny rozproszenia.
- Raport końcowy z wykresem skupienia.

## Jak uruchomić projekt

1. **Sklonuj repozytorium:**
   ```bash
   git clone https://github.com/a-wierzba/AsystentSkupieniaAI.git
   ```
2. **Wejdź do katalogu projektu:**
   ```bash
   cd AsystentSkupieniaAI
   ```
3. **Zainstaluj biblioteki:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Uruchom aplikację** (modele AI pobiorą się automatycznie przy pierwszym uruchomieniu):
   ```bash
   python main.py
   ```

## Jak działa sesja skupienia?

1. **Faza Kalibracji (30 sek)** — Uczeń przygotowuje stanowisko. Na ekranie widać podgląd z kamery i ramki AI. Można zacząć wcześniej, naciskając klawisz **S**.

2. **Faza Pracy (5 min)** — Okno kamery znika, aby nie rozpraszać ucznia. AI pracuje w tle, analizując skupienie (kąt nachylenia głowy i emocje) oraz zbierając dane do raportu.

3. **Faza Raportu** — Po 5 minutach okno powraca z komunikatem o końcu sesji. Naciśnięcie klawisza **R** generuje interaktywny wykres „Raport Uważności”.

Projekt idealnie nadaje się do przeprowadzenia szybkich, 5‑minutowych sesji testowych podczas lekcji o AI.

## Autor
Projekt przygotowany przez: [Andrzej W.](https://github.com/a-wierzba)
