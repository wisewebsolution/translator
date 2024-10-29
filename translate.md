# translation_app.py
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
import pickle
import queue
from concurrent.futures import ThreadPoolExecutor

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translator.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TranslationJob:
    text: str
    source_col: str
    row_index: int
    target_lang: str
    use_product_naming: bool

@dataclass
class ExecutionState:
    current_row: int
    processed_rows: Dict[str, list]
    timestamp: float
    batch_results: Dict[str, str]
    api_usage: Dict[str, int]
    
    def to_json(self):
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

class TokenManager:
    def __init__(self):
        self.MAX_CONTEXT_TOKENS = 200000
        self.MAX_OUTPUT_TOKENS = 4096
        self.tokenizer = anthropic.Tokenizer()
    
    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def validate_input(self, text: str, max_output: int = None) -> tuple[bool, str]:
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens >= self.MAX_CONTEXT_TOKENS:
            return False, f"Tekst przekracza limit {self.MAX_CONTEXT_TOKENS} tokenów"
        if max_output and max_output > self.MAX_OUTPUT_TOKENS:
            return False, f"Limit wyjścia nie może przekraczać {self.MAX_OUTPUT_TOKENS} tokenów"
        return True, ""

class PromptManager:
    def __init__(self):
        self.prompts = {
            'translation': {
                'default': """You are a professional translator with expertise in many languages.
                Task: Translate the provided text while preserving:
                - Technical terminology
                - Formatting and special characters
                - Brand names (leave unchanged)
                - Tone and context
                """
            },
            'product_naming': {
                'default': """You are a creative product naming specialist.
                Task: Generate memorable product names (2-4 words) based on descriptions.
                Requirements:
                - Create unique, market-ready names
                - Ensure names are memorable and easy to pronounce
                - Consider target market and industry
                """
            }
        }
        
    def get_prompt(self, prompt_type: str, custom_instructions: Optional[str] = None) -> str:
        base_prompt = self.prompts.get(prompt_type, {}).get('default', '')
        if custom_instructions:
            return f"{base_prompt}\n\nCustom Instructions:\n{custom_instructions}"
        return base_prompt
        
    def save_custom_prompt(self, name: str, prompt: str):
        if 'custom_prompts' not in self.prompts:
            self.prompts['custom_prompts'] = {}
        self.prompts['custom_prompts'][name] = prompt

class TranslationCache:
    def __init__(self, cache_dir: str = ".translation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, text: str, target_lang: str) -> str:
        return hashlib.md5(f"{text}:{target_lang}".encode()).hexdigest()
        
    def get(self, text: str, target_lang: str) -> Optional[str]:
        cache_key = self._get_cache_key(text, target_lang)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with cache_file.open('r') as f:
                    cache_data = json.load(f)
                if time.time() - cache_data['timestamp'] < 300:  # 5 minut
                    return cache_data['translation']
            except Exception as e:
                logging.error(f"Cache read error: {e}")
        return None
        
    def set(self, text: str, target_lang: str, translation: str):
        cache_key = self._get_cache_key(text, target_lang)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with cache_file.open('w') as f:
                json.dump({
                    'text': text,
                    'target_lang': target_lang,
                    'translation': translation,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logging.error(f"Cache write error: {e}")

class APIUsageTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        
    def calculate_cost(self) -> Dict[str, float]:
        COSTS = {
            'base_input': 3.0 / 1_000_000,
            'cache_write': 3.75 / 1_000_000,
            'cache_read': 0.30 / 1_000_000,
            'output': 15.0 / 1_000_000,
        }
        
        return {
            'input_cost': self.input_tokens * COSTS['base_input'],
            'cache_write_cost': self.cache_write_tokens * COSTS['cache_write'],
            'cache_read_cost': self.cache_read_tokens * COSTS['cache_read'],
            'output_cost': self.output_tokens * COSTS['output'],
            'total_cost': (self.input_tokens * COSTS['base_input'] +
                         self.cache_write_tokens * COSTS['cache_write'] +
                         self.cache_read_tokens * COSTS['cache_read'] +
                         self.output_tokens * COSTS['output'])
        }

class BatchProcessor:
    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size
        self.queue = queue.Queue()
        self.results = {}
        self.is_processing = False
        
    def add_job(self, job: TranslationJob):
        self.queue.put(job)
        
    def process_batch(self, processor) -> List[Dict]:
        batch = []
        batch_indices = []
        
        while len(batch) < self.batch_size and not self.queue.empty():
            job = self.queue.get()
            batch.append(job)
            batch_indices.append((job.row_index, job.source_col))
            
        if not batch:
            return []
            
        texts = [job.text for job in batch]
        target_lang = batch[0].target_lang
        use_product_naming = batch[0].use_product_naming
        
        translations = processor.process_batch(texts, target_lang, 
                                            st.session_state.temperature,
                                            use_product_naming)
        
        results = []
        for job, translation in zip(batch, translations):
            results.append({
                'row_index': job.row_index,
                'source_col': job.source_col,
                'translation': translation
            })
            
        return results

class TranslationProcessor:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.MODEL = "claude-3-5-sonnet-20241022"
        self.tracker = APIUsageTracker()
        self.cache = TranslationCache()
        self.prompt_manager = PromptManager()
        self.batch_processor = BatchProcessor()
        self.token_manager = TokenManager()
        self.cache_enabled = True
        
    def process_batch(self, texts: List[str], target_lang: str, 
                     temperature: float, use_product_naming: bool = False) -> List[str]:
        # Walidacja wejścia
        for text in texts:
            is_valid, error = self.token_manager.validate_input(text)
            if not is_valid:
                raise ValueError(f"Błąd walidacji tekstu: {error}")
        
        # Sprawdzanie cache
        results = []
        texts_to_translate = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cached_translation = self.cache.get(text, target_lang)
            if cached_translation and self.cache_enabled:
                results.append(cached_translation)
            else:
                texts_to_translate.append(text)
                cache_indices.append(i)
                
        if texts_to_translate:
            try:
                prompt_type = 'product_naming' if use_product_naming else 'translation'
                system_prompt = self.prompt_manager.get_prompt(prompt_type)
                
                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=2000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "\n---\n".join(texts_to_translate),
                                "cache_control": {"type": "ephemeral"} if self.cache_enabled else None
                            }
                        ]
                    }],
                    extra_headers={
                        "anthropic-beta": "prompt-caching-2024-07-31,message-batches-2024-09-24"
                    }
                )
                
                self.update_stats(response)
                new_translations = response.content[0].text.split('---')
                new_translations = [t.strip() for t in new_translations]
                
                # Aktualizacja cache
                for text, translation in zip(texts_to_translate, new_translations):
                    self.cache.set(text, target_lang, translation)
                
                # Łączenie wyników
                final_results = results.copy()
                for i, translation in zip(cache_indices, new_translations):
                    final_results.insert(i, translation)
                    
                return final_results
                
            except Exception as e:
                logging.error(f"Translation error: {str(e)}")
                raise
                
        return results

    def update_stats(self, response):
        self.tracker.input_tokens += response.usage.input_tokens
        self.tracker.output_tokens += response.usage.output_tokens
        if hasattr(response.usage, 'cache_creation_input_tokens'):
            self.tracker.cache_write_tokens += response.usage.cache_creation_input_tokens
        if hasattr(response.usage, 'cache_read_input_tokens'):
            self.tracker.cache_read_tokens += response.usage.cache_read_input_tokens

class ExecutionController:
    def __init__(self, save_dir: str = ".translation_state"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.current_state: Optional[ExecutionState] = None
        self.is_paused = False
        
    def save_state(self, state: ExecutionState, session_id: str):
        state_file = self.save_dir / f"state_{session_id}.json"
        with state_file.open('w') as f:
            f.write(state.to_json())
            
    def load_state(self, session_id: str) -> Optional[ExecutionState]:
        state_file = self.save_dir / f"state_{session_id}.json"
        if state_file.exists():
            with state_file.open('r') as f:
                return ExecutionState.from_json(f.read())
        return None
        
    def list_saved_states(self) -> List[Dict]:
        states = []
        for state_file in self.save_dir.glob("state_*.json"):
            try:
                with state_file.open('r') as f:
                    state = ExecutionState.from_json(f.read())
                    states.append({
                        'session_id': state_file.stem.replace('state_', ''),
                        'timestamp': datetime.fromtimestamp(state.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'processed_rows': sum(len(rows) for rows in state.processed_rows.values()),
                        'columns': list(state.processed_rows.keys())
                    })
            except Exception as e:
                logging.error(f"Error loading state {state_file}: {e}")
        return sorted(states, key=lambda x: x['timestamp'], reverse=True)

# UI Components
def render_sidebar():
    with st.sidebar:
        st.subheader("📊 Zużycie API")
        costs = st.session_state.processor.tracker.calculate_cost()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Całkowity koszt", f"${costs['total_cost']:.4f}")
            st.metric("Koszt wejścia", f"${costs['input_cost']:.4f}")
        with col2:
            st.metric("Koszt wyjścia", f"${costs['output_cost']:.4f}")
            if st.session_state.processor.cache_enabled:
                st.metric("Oszczędności cache", 
                         f"${(costs['cache_read_cost'] - costs['cache_write_cost']):.4f}")

def render_configuration():
    with st.expander("⚙️ Konfiguracja"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 1.0, 0.3, 0.1,
                help="Wyższa wartość = bardziej kreatywne tłumaczenia"
            )
        with col2:
            st.session_state.processor.batch_size = st.number_input(
                "Rozmiar batch", 1, 10, 2,
                help="Liczba tekstów przetwarzanych jednocześnie"
            )
        with col3:
            st.session_state.processor.cache_enabled = st.checkbox(
                "Użyj cache'owania", True,
                help="Przyspiesza tłumaczenie powtarzających się tekstów"
            )
            
    with st.expander("📝 Konfiguracja promptu"):
        prompt_text = st.text_area(
            "Edytuj prompt",
            value=st.session_state.get('current_prompt', 
                st.session_state.processor.prompt_manager.get_prompt('translation')),
            height=200,
            help="Edytuj prompt używany do tłumaczenia."
        )
        
        col1, col2 = st.columns(2)
  with col1:
                max_output_tokens = st.number_input(
                    "Max output tokens",
                    min_value=1,
                    max_value=4096,
                    value=1000,
                    help="Maksymalna długość odpowiedzi w tokenach"
                )
            
            with col2:
                if st.button("Zapisz prompt"):
                    try:
                        is_valid, error = validate_prompt(prompt_text)
                        if is_valid:
                            st.session_state['current_prompt'] = prompt_text
                            st.success("Prompt zapisany!")
                        else:
                            st.error(f"Błąd w prompcie: {error}")
                    except Exception as e:
                        st.error(f"Błąd podczas zapisywania promptu: {str(e)}")

def render_execution_controls():
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Kontrola wykonania")
    
    if 'is_processing' in st.session_state and st.session_state.is_processing:
        if st.sidebar.button("⏸️ Pauza"):
            st.session_state.controller.is_paused = True
            st.session_state.processor.save_current_state()
            st.sidebar.success("Przetwarzanie zatrzymane")
            
        current_progress = st.session_state.get('processed_rows', 0)
        total_rows = st.session_state.get('total_rows', 0)
        if total_rows > 0:
            st.sidebar.progress(current_progress / total_rows)
            st.sidebar.text(f"Postęp: {current_progress}/{total_rows}")
    
    saved_states = st.session_state.controller.list_saved_states()
    if saved_states:
        st.sidebar.subheader("💾 Zapisane sesje")
        selected_state = st.sidebar.selectbox(
            "Wybierz sesję do wznowienia",
            options=saved_states,
            format_func=lambda x: f"{x['timestamp']} ({x['processed_rows']} wierszy)"
        )
        
        if selected_state and st.sidebar.button("▶️ Wznów"):
            load_and_resume_translation(selected_state['session_id'])

def main():
    st.set_page_config(page_title="Advanced CSV Translator", layout="wide")
    
    st.title("🌐 Zaawansowany Translator CSV z Claude 3.5")
    
    # Inicjalizacja stanu
    if 'processor' not in st.session_state:
        st.session_state.processor = TranslationProcessor()
        st.session_state.temperature = 0.3
        st.session_state.controller = ExecutionController()
    
    # Renderowanie komponentów
    render_sidebar()
    render_configuration()
    render_execution_controls()
    
    # Upload pliku
    uploaded_file = st.file_uploader("Wybierz plik CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Podgląd danych:")
            st.dataframe(df.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                columns_to_translate = st.multiselect(
                    "Wybierz kolumny do tłumaczenia",
                    df.columns
                )
            with col2:
                target_language = st.selectbox(
                    "Język docelowy",
                    ['English', 'German', 'French', 'Spanish', 'Italian', 'Polish']
                )
            with col3:
                use_product_naming = st.checkbox(
                    "Użyj Product Naming Pro", False,
                    help="Generuje kreatywne nazwy produktów zamiast tłumaczenia"
                )
            
            # Test tłumaczenia
            if st.button("🔬 Test tłumaczenia"):
                if columns_to_translate:
                    test_text = df[columns_to_translate[0]].iloc[0]
                    with st.spinner("Tłumaczenie testowe..."):
                        translation = st.session_state.processor.process_batch(
                            [test_text], target_language, st.session_state.temperature,
                            use_product_naming)[0]
                    
                    st.write("Wynik testu:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Oryginał:")
                        st.code(test_text)
                    with col2:
                        st.write("Tłumaczenie:")
                        st.code(translation)
            
            # Rozpoczęcie tłumaczenia
            if st.button("� Rozpocznij tłumaczenie"):
                st.session_state.is_processing = True
                st.session_state.df = df
                process_translations_with_control(
                    df, columns_to_translate, target_language, use_product_naming
                )
            
        except Exception as e:
            st.error(f"Wystąpił błąd: {str(e)}")
            logging.error("Application error", exc_info=True)
            
            if 'temp_results' in st.session_state:
                st.warning("Znaleziono tymczasowe wyniki. Możesz je pobrać poniżej:")
                st.download_button(
                    label="⚠️ Pobierz ostatnie wyniki",
                    data=st.session_state['temp_results'],
                    file_name=f"recovered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Wystąpił krytyczny błąd aplikacji")
        logging.critical("Application crash", exc_info=True)