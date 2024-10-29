import streamlit as st
from deep_translator import GoogleTranslator
import pandas as pd

def translate_text(text, source_lang='auto', target_lang='pl'):
    """
    Tłumaczy tekst z jednego języka na drugi używając Google Translate.
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return f"Błąd podczas tłumaczenia: {str(e)}"

def main():
    st.title("🌍 Translator Tekstów")
    st.write("Aplikacja do tłumaczenia tekstów między różnymi językami")

    # Wybór trybu tłumaczenia
    mode = st.radio(
        "Wybierz tryb tłumaczenia:",
        ["Pojedynczy tekst", "Plik CSV"]
    )

    if mode == "Pojedynczy tekst":
        # Pole tekstowe do wprowadzenia tekstu
        text = st.text_area("Wprowadź tekst do przetłumaczenia:", height=150)
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.text_input(
                "Kod języka źródłowego (lub zostaw puste dla autodetekcji):",
                value="auto"
            )
        
        with col2:
            target_lang = st.text_input(
                "Kod języka docelowego (np. pl, en, de):",
                value="pl"
            )

        if st.button("Tłumacz"):
            if text:
                with st.spinner("Tłumaczenie..."):
                    result = translate_text(text, source_lang, target_lang)
                st.success("Tłumaczenie zakończone!")
                st.write("### Wynik tłumaczenia:")
                st.write(result)
            else:
                st.warning("Wprowadź tekst do przetłumaczenia!")

    else:  # Tryb CSV
        uploaded_file = st.file_uploader("Wybierz plik CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("### Podgląd danych:")
            st.dataframe(df.head())

            # Wybór kolumn do tłumaczenia
            columns = st.multiselect(
                "Wybierz kolumny do przetłumaczenia:",
                df.columns
            )

            target_lang = st.text_input(
                "Kod języka docelowego (np. pl, en, de):",
                value="pl"
            )

            if st.button("Tłumacz CSV"):
                if columns:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Tłumaczenie wybranych kolumn
                    for idx, col in enumerate(columns):
                        status_text.text(f"Tłumaczenie kolumny: {col}")
                        df[f"{col}_translated"] = df[col].apply(
                            lambda x: translate_text(str(x), 'auto', target_lang)
                        )
                        progress = (idx + 1) / len(columns)
                        progress_bar.progress(progress)

                    progress_bar.progress(100)
                    status_text.text("Tłumaczenie zakończone!")

                    # Zapisz wyniki
                    st.write("### Przetłumaczone dane:")
                    st.dataframe(df)

                    # Przycisk do pobrania wyników
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Pobierz przetłumaczone dane",
                        data=csv,
                        file_name="translated_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Wybierz kolumny do przetłumaczenia!")

    # Informacje o kodach języków
    with st.expander("📚 Popularne kody języków"):
        st.write("""
        - 'en' - angielski
        - 'pl' - polski
        - 'de' - niemiecki
        - 'es' - hiszpański
        - 'fr' - francuski
        - 'it' - włoski
        - 'ru' - rosyjski
        - 'uk' - ukraiński
        - 'cs' - czeski
        - 'sk' - słowacki
        """)

if __name__ == "__main__":
    main()
