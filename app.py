import streamlit as st
import pandas as pd
import anthropic
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import queue
import hashlib
import os

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translator.log'),
        logging.StreamHandler()
    ]
)

class TranslationProcessor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.MODEL = "claude-3-5-sonnet-20241022"
        
    def translate_text(self, text: str, target_lang: str, temperature: float = 0.3, 
                      max_tokens: int = 2000, custom_prompt: str = None, 
                      use_product_naming: bool = False) -> str:
        try:
            system_prompt = custom_prompt if custom_prompt else (
                "You are a creative product naming specialist. Generate memorable product names (2-4 words) "
                "that are unique, market-ready, and easy to pronounce for the target market."
                if use_product_naming else
                f"You are a professional translator. Translate the following text to {target_lang}. "
                "Preserve all formatting and special characters."
            )
            
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": text
                }]
            )
            return message.content[0].text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            raise

def main():
    st.set_page_config(page_title="CSV Translator", layout="wide")
    
    st.title("🌐 CSV Translator with Claude 3.5")
    
    # API Key Input
    api_key = st.text_input("Enter your Anthropic API Key:", type="password")
    if not api_key:
        st.warning("Please enter your Anthropic API Key to proceed.")
        return
    
    # Initialize processor
    if 'processor' not in st.session_state or st.session_state.get('api_key') != api_key:
        try:
            st.session_state.processor = TranslationProcessor(api_key)
            st.session_state.api_key = api_key
        except Exception as e:
            st.error(f"Error initializing translator: {str(e)}")
            return
    
    # Upload pliku
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                columns_to_translate = st.multiselect(
                    "Select columns to translate",
                    df.columns
                )
            with col2:
                target_language = st.selectbox(
                    "Target language",
                    ['Czech', 'English']
                )
            
            # Advanced Settings
            with st.expander("⚙️ Advanced Settings"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Higher values make output more creative, lower values more deterministic"
                )
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4096,
                    value=2000,
                    help="Maximum number of tokens in the response"
                )
                
                custom_prompt = st.text_area(
                    "Custom System Prompt",
                    help="Override default system prompt (leave empty for default)",
                    placeholder="Enter custom prompt here..."
                )
                
                use_product_naming = st.checkbox(
                    "Use Product Naming Pro",
                    help="Generate creative product names instead of direct translation"
                )
            
            # Test tłumaczenia
            if st.button("🔬 Test translation"):
                if columns_to_translate:
                    test_text = df[columns_to_translate[0]].iloc[0]
                    with st.spinner("Translating test text..."):
                        try:
                            translation = st.session_state.processor.translate_text(
                                test_text, 
                                target_language,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                custom_prompt=custom_prompt if custom_prompt.strip() else None,
                                use_product_naming=use_product_naming
                            )
                            
                            st.write("Test results:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Original:")
                                st.code(test_text)
                            with col2:
                                st.write("Translation:")
                                st.code(translation)
                        except Exception as e:
                            st.error(f"Translation error: {str(e)}")
            
            # Rozpoczęcie tłumaczenia
            if st.button("🚀 Start translation"):
                if not columns_to_translate:
                    st.warning("Please select columns to translate.")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_rows = len(df) * len(columns_to_translate)
                processed_rows = 0
                
                try:
                    for col in columns_to_translate:
                        status_text.text(f"Processing column: {col}")
                        
                        df[f"{col}_translated"] = ""
                        for idx, text in enumerate(df[col]):
                            translation = st.session_state.processor.translate_text(
                                str(text), 
                                target_language,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                custom_prompt=custom_prompt if custom_prompt.strip() else None,
                                use_product_naming=use_product_naming
                            )
                            df.at[idx, f"{col}_translated"] = translation
                            
                            processed_rows += 1
                            progress = processed_rows / total_rows
                            progress_bar.progress(progress)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Translation completed!")
                    
                    # Display results
                    st.write("### Translated data:")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download translated data",
                        data=csv,
                        file_name=f"translated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during translation: {str(e)}")
                    logging.error("Translation error", exc_info=True)
                    
                    # Save temporary results
                    csv = df.to_csv(index=False)
                    st.warning("Found temporary results. You can download them below:")
                    st.download_button(
                        label="⚠️ Download partial results",
                        data=csv,
                        file_name=f"partial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logging.error("Application error", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical application error")
        logging.critical("Application crash", exc_info=True)
